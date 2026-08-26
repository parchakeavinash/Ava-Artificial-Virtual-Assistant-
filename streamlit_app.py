import logging
import queue
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from app.config.settings import settings
from app.voice.runner import AgentRunner
from app.voice.stt import SarvamRealtimeSTT

logging.basicConfig(level=logging.INFO)

# =========================================================
# PAGE CONFIG & TITLE
# =========================================================
st.set_page_config(
    page_title="Ava - Voice AI Assistant",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Ava - Voice AI Assistant")
st.caption("Powered by Sarvam AI (STT + TTS), Google Gemini, and Real-Time Tools")

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
if "stt_client" not in st.session_state:
    st.session_state.stt_client = SarvamRealtimeSTT()
    st.session_state.stt_client.start()

if "agent_runner" not in st.session_state:
    st.session_state.agent_runner = AgentRunner()
    st.session_state.agent_runner.start()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "partial_text" not in st.session_state:
    st.session_state.partial_text = ""

if "pending_utterance" not in st.session_state:
    st.session_state.pending_utterance = ""

if "active_audio" not in st.session_state:
    st.session_state.active_audio = None


# =========================================================
# AUDIO PROCESSOR CALLBACK FOR WEBRTC
# =========================================================
def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    """Extract mono PCM16 audio and send to Sarvam STT."""
    try:
        sound = frame.to_ndarray()
        if sound.ndim > 1:
            sound = sound.mean(axis=0)
        sound_int16 = sound.astype(np.int16)
        st.session_state.stt_client.push_audio(sound_int16.tobytes())
    except Exception:
        pass
    return frame


# =========================================================
# SIDEBAR CONTROLS
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings & Tools")
    st.markdown(f"**Model:** `{settings.GEMINI_MODEL}`")
    st.markdown("**Voice:** `shubh (bulbul:v3)`")
    st.markdown("**Tools Enabled:**")
    st.markdown("- 🔍 Firecrawl Web Search")
    st.markdown("- ✉️ Gmail (Send, Read, Delete with Confirmation)")

    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history.clear()
        st.session_state.agent_runner.reset_conversation()
        st.session_state.active_audio = None
        st.session_state.partial_text = ""
        st.session_state.pending_utterance = ""
        st.rerun()


# =========================================================
# WEBRTC MICROPHONE STREAMER
# =========================================================
webrtc_streamer(
    key="ava-mic",
    mode=WebRtcMode.SENDONLY,
    audio_frame_callback=audio_frame_callback,
    media_stream_constraints={"video": False, "audio": True},
)


# =========================================================
# LIVE CONVERSATION PANEL (AUTO-RERUN FRAGMENT)
# =========================================================
@st.fragment(run_every=0.3)
def conversation_panel():
    stt = st.session_state.stt_client
    agent = st.session_state.agent_runner

    # 1. Drain STT events
    events = stt.get_events()
    for event in events:
        event_type = event.get("type")
        if event_type == "partial":
            st.session_state.partial_text = event.get("text", "")
        elif event_type == "final":
            text = event.get("text", "").strip()
            if text:
                if st.session_state.pending_utterance:
                    st.session_state.pending_utterance += " " + text
                else:
                    st.session_state.pending_utterance = text
                agent.push_transcript(text)
            st.session_state.partial_text = ""
        elif event_type == "error":
            st.error(f"STT Error: {event.get('message', 'Unknown error')}")

    # 2. Drain Agent results
    new_results = agent.get_results()
    for r in new_results:
        user_text = st.session_state.pending_utterance
        st.session_state.pending_utterance = ""

        st.session_state.chat_history.append({
            "user": user_text,
            "ava": r.get("text", ""),
            "error": r.get("error", False),
        })

        if r.get("audio_bytes") and not r.get("error"):
            st.session_state.active_audio = r["audio_bytes"]

    # 3. Render Chat Dialogue
    st.divider()
    st.subheader("💬 Live Conversation")

    has_content = (
        bool(st.session_state.chat_history)
        or bool(st.session_state.pending_utterance)
        or bool(st.session_state.partial_text)
    )

    if not has_content:
        st.caption("🎙️ Start speaking to chat with Ava...")

    for turn in st.session_state.chat_history:
        if turn.get("user"):
            st.markdown(f"🗣️ **You:** {turn['user']}")
        if turn.get("ava"):
            st.markdown(f"🤖 **Ava:** {turn['ava']}")
        if turn.get("error"):
            st.warning(turn["ava"])

    if st.session_state.pending_utterance:
        st.markdown(f"🗣️ **You:** {st.session_state.pending_utterance}")

    if st.session_state.partial_text:
        st.markdown(f"🎙️ *Listening:* {st.session_state.partial_text}")

    # Live busy status
    if getattr(agent, "is_busy", False):
        st.info(f"🔎 {getattr(agent, 'current_action', 'Ava is thinking & using tools...')}")

    # Persistent Audio Player
    if st.session_state.active_audio:
        st.audio(
            st.session_state.active_audio,
            format="audio/wav",
            autoplay=True,
        )


# Run Panel
conversation_panel()
