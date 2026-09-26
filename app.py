import os
import asyncio
import io
import requests
import streamlit as st
from gtts import gTTS
from dotenv import load_dotenv
from streamlit_mic_recorder import speech_to_text

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load environment variables from local .env file
load_dotenv()

# ==========================================
# 1. TOOL DEFINITIONS
# ==========================================

def serper_search_tool(query: str) -> str:
    """
    Searches Google using the Serper API and returns relevant snippet results.
    """
    api_key = os.getenv("SERPER_API_KEY") or (st.secrets.get("SERPER_API_KEY") if hasattr(st, "secrets") else None)
    if not api_key or "your-actual" in api_key:
        return "Error: Valid SERPER_API_KEY missing from environment variables or .env file."

    url = "https://google.serper.dev/search"
    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}
    payload = {"q": query, "num": 3}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        results = response.json()
        organic = results.get("organic", [])
        if not organic:
            return "No web search results found for this query."

        formatted_snippets = []
        for item in organic:
            title = item.get("title", "No Title")
            snippet = item.get("snippet", "No Snippet")
            link = item.get("link", "")
            formatted_snippets.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}")

        return "\n\n".join(formatted_snippets)
    except Exception as e:
        return f"Serper API search failed: {str(e)}"


def save_to_file_tool(query: str, assistant_answer: str, web_search_answer: str) -> str:
    """
    Appends current user query, Agent 1 answer, and Agent 2 answer to answers.txt.
    Keeps all previous outputs saved cleanly above it.
    """
    file_path = "answers.txt"
    log_entry = (
        f"==================================================\n"
        f"📌 USER QUERY:\n{query}\n\n"
        f"💡 AGENT 1 (Direct Knowledge Answer):\n{assistant_answer}\n\n"
        f"🌐 AGENT 2 (Web Search Answer):\n{web_search_answer}\n"
        f"==================================================\n\n"
    )
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
        return f"Successfully saved session history to {file_path}."
    except Exception as e:
        return f"Failed to write to file: {str(e)}"


# ==========================================
# 2. SEQUENTIAL MULTI-AGENT WORKFLOW
# ==========================================

async def execute_customer_support_team(user_query: str, status_container):
    """
    Assigns the initial task and streams through Agent 1 -> Agent 2 -> Agent 3 automatically.
    """
    openai_key = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
    if not openai_key:
        st.error("Missing `OPENAI_API_KEY`. Please set it in your environment variables or .env file.")
        return None, None

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=openai_key)

    # Agent 1: Direct Knowledge Base
    assistant = AssistantAgent(
        name="Assistant",
        model_client=model_client,
        system_message=(
            "You are Agent 1 (Direct Knowledge Base). Answer the customer query directly using "
            "your internal knowledge without using external tools. Keep your answer clear and accurate."
        )
    )

    # Agent 2: Web Search Assistant
    web_assistant = AssistantAgent(
        name="Web_Search_Assistant",
        model_client=model_client,
        tools=[serper_search_tool],
        system_message=(
            "You are Agent 2 (Web Search Assistant). You MUST invoke serper_search_tool using the user's "
            "original query. Once you receive search snippets from the tool, summarize them into a clean answer."
        )
    )

    # Agent 3: Entry Agent (Logging)
    entry_agent = AssistantAgent(
        name="Entry_Agent",
        model_client=model_client,
        tools=[save_to_file_tool],
        system_message=(
            "You are Agent 3 (Logging Agent). Extract the original query, the answer from 'Assistant', "
            "and the answer from 'Web_Search_Assistant'. You MUST call save_to_file_tool with these inputs. "
            "After calling the tool, reply with 'Logging complete.' and end your message with 'TERMINATE'."
        )
    )

    # Sequential Group Chat (Strict 1 -> 2 -> 3 Round Robin)
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=10)
    team = RoundRobinGroupChat(
        participants=[assistant, web_assistant, entry_agent],
        termination_condition=termination
    )

    direct_ans, web_ans = "", ""

    status_container.write("🔄 **Step 1/3:** Agent 1 (Assistant) processing query...")

    # Streaming Agent Turns automatically one by one
    async for message in team.run_stream(task=user_query):
        source = getattr(message, "source", None)

        # Agent 1 Complete
        if source == "Assistant" and isinstance(message, TextMessage):
            direct_ans = message.content
            status_container.write("✅ **Step 1 Complete:** Direct Answer generated.")
            status_container.write("🔄 **Step 2/3:** Agent 2 (Web Search Assistant) searching web...")

        # Agent 2 Complete
        elif source == "Web_Search_Assistant" and isinstance(message, TextMessage):
            web_ans = message.content
            status_container.write("✅ **Step 2 Complete:** Web Search data retrieved.")
            status_container.write("🔄 **Step 3/3:** Agent 3 (Entry Agent) logging session to `answers.txt`...")

        # Agent 3 Complete
        elif source == "Entry_Agent":
            status_container.write("✅ **Step 3 Complete:** Entry Agent saved log to `answers.txt`!")

    return direct_ans, web_ans


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def text_to_speech(text: str) -> io.BytesIO:
    """Converts input text into an in-memory MP3 audio stream."""
    tts = gTTS(text=text, lang="en")
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp


# ==========================================
# 4. STREAMLIT INTERFACE & VOICE SIDEBAR
# ==========================================

st.set_page_config(page_title="Voice & Text Multi-Agent Support", layout="wide")

st.title("🤖 Multi-Agent Customer Support System")
st.markdown("Automatic Pipeline: **Input** $\rightarrow$ **Agent 1** $\rightarrow$ **Agent 2** $\rightarrow$ **Agent 3 (Saved to File)**")

# Initialize Session State
if "query_text" not in st.session_state:
    st.session_state.query_text = ""
if "trigger_run" not in st.session_state:
    st.session_state.trigger_run = False

# ------------------------------------------
# SIDEBAR: VOICE ASSISTANT
# ------------------------------------------
with st.sidebar:
    st.header("🎙️ Voice Assistant Panel")
    st.info("Click **Start Recording**, speak your query, then click **Stop**.")
    
    spoken_text = speech_to_text(
        language="en",
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop & Process",
        key="voice_recorder"
    )

    if spoken_text:
        st.success(f"**Transcribed Speech:** \"{spoken_text}\"")
        st.session_state.query_text = spoken_text
        st.session_state.trigger_run = True

# ------------------------------------------
# MAIN DISPLAY: TEXT INPUT & AGENT DISPLAY
# ------------------------------------------
user_query = st.text_area(
    "Customer Support Query:",
    value=st.session_state.query_text,
    placeholder="e.g., How do I reset my password?",
    height=100
)

submit_button = st.button("Submit Query", type="primary")

if submit_button:
    st.session_state.query_text = user_query
    st.session_state.trigger_run = True

# Automatic Sequential Execution
if st.session_state.trigger_run and st.session_state.query_text.strip():
    st.session_state.trigger_run = False
    active_query = st.session_state.query_text.strip()

    # Expandable status showing step-by-step progress
    with st.status("🚀 Executing 3-Agent Sequential Workflow...", expanded=True) as status:
        direct_answer, web_answer = asyncio.run(
            execute_customer_support_team(active_query, status)
        )
        status.update(label="🎉 Workflow Complete!", state="complete", expanded=False)

    if direct_answer or web_answer:
        st.success("Query and answers successfully logged to `answers.txt`!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💡 Agent 1: Direct Knowledge Base")
            st.info(direct_answer if direct_answer else "No direct answer generated.")

        with col2:
            st.subheader("🌐 Agent 2: Web Search Assistant")
            st.success(web_answer if web_answer else "No web search answer generated.")

        # Voice audio output playback in sidebar
        if direct_answer:
            with st.sidebar:
                st.subheader("🔊 Audio Response")
                audio_buffer = text_to_speech(direct_answer)
                st.audio(audio_buffer, format="audio/mp3", autoplay=True)

# ------------------------------------------
# AGENT 3 FILE VIEWER & MANAGEMENT (`answers.txt`)
# ------------------------------------------
st.markdown("---")
col_log, col_clear = st.columns([4, 1])

with col_log:
    st.subheader("📁 Saved Log File (`answers.txt`)")

with col_clear:
    if st.button("🗑️ Clear History", help="Clears all entries in answers.txt"):
        if os.path.exists("answers.txt"):
            open("answers.txt", "w", encoding="utf-8").close()
            st.toast("Cleared `answers.txt` history!")
            st.rerun()

with st.expander("Click to view full log history (Current + Previous Outputs)", expanded=True):
    if os.path.exists("answers.txt"):
        with open("answers.txt", "r", encoding="utf-8") as f:
            file_contents = f.read()
        if file_contents.strip():
            st.code(file_contents, language="text")
        else:
            st.info("`answers.txt` exists but is currently empty.")
    else:
        st.warning("`answers.txt` has not been created yet. Submit a query to generate it.")