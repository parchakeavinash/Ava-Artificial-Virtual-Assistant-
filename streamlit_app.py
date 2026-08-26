import logging
import queue
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from app.agent.langchain_agent import agent_metrics
from app.config.settings import settings
from app.voice.runner import AgentRunner
from app.voice.scheduler import drain_reminders, start_scheduler
from app.voice.stt import SarvamRealtimeSTT
from app.voice.whisper_stt import GroqWhisperSTT, tracker as whisper_tracker

logging.basicConfig(level=logging.INFO)

# =========================================================
# PAGE CONFIG & TITLE
# =========================================================
st.set_page_config(
    page_title="Ava - Multi-LLM Voice AI Assistant",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Ava - Voice AI Assistant")
st.caption("Powered by Sarvam AI (STT + TTS), Google Gemini, and Real-Time Tools")

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

# --- Reminder Scheduler Config ---
REMINDER_HOUR   = 9   # 9 AM local time
REMINDER_MINUTE = 0

if "stt_client" not in st.session_state:
    st.session_state.stt_client = SarvamRealtimeSTT()
    st.session_state.stt_client.start()

if "whisper_stt" not in st.session_state:
    st.session_state.whisper_stt = GroqWhisperSTT()

if "agent_runner" not in st.session_state:
    st.session_state.agent_runner = AgentRunner()
    st.session_state.agent_runner.start()

# Start the background reminder scheduler exactly ONCE per server session.
# We store a flag in session_state so multiple browser tabs don't start
# multiple scheduler threads.
if "scheduler_started" not in st.session_state:
    start_scheduler(reminder_hour=REMINDER_HOUR, reminder_minute=REMINDER_MINUTE)
    st.session_state.scheduler_started = True

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
    """Extract mono PCM16 audio and send to STT."""
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
# SIDEBAR CONTROLS & LIVE QUOTA METRICS
# =========================================================
with st.sidebar:
    st.header("⚙️ Live Engine & Quota Monitor")

    # 1. LLM Status
    st.subheader("🧠 Multi-LLM Routing")
    metrics = agent_metrics.get_stats()
    st.success(f"**Active LLM:** `{metrics['last_provider']}`")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Groq Calls", metrics["groq_calls"])
    with col2:
        st.metric("Gemini Fallbacks", metrics["gemini_fallback_calls"])

    st.divider()

    # 2. Whisper STT Status & Quota
    st.subheader("🎙️ Groq Whisper STT")
    w_stats = whisper_tracker.get_stats()
    st.info(f"**STT Model:** `{w_stats['active_model']}`")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("RPM Left", f"{w_stats['rpm_remaining']}/{w_stats['rpm_limit']}")
    with c2:
        st.metric("Daily Left", f"{w_stats['rpd_remaining']}/{w_stats['rpd_limit']}")

    st.caption("ℹ️ Auto-switches to `whisper-large-v3-turbo` if limits or errors occur.")

    st.divider()

    # 3. Enabled Tools
    st.subheader("🛠️ Active Tools")
    st.markdown("- 🔍 **Web Search** (Firecrawl)")
    st.markdown("- ✉️ **Gmail** (Send, Read, Delete with Confirmation)")
    st.markdown("- 📝 **Notion** (Pages, Notes, Databases)")
    st.markdown("- ✅ **Task Manager** (Create, List, Complete)")
    st.markdown("- 📓 **Diary / Ideas** (Save, Search, Read)")

    st.divider()

    # 4. Proactive Reminder Scheduler
    st.subheader("⏰ Daily Reminder")
    import datetime as _dt
    next_reminder = _dt.datetime.now().replace(
        hour=REMINDER_HOUR, minute=REMINDER_MINUTE, second=0, microsecond=0
    )
    if next_reminder < _dt.datetime.now():
        next_reminder = next_reminder + _dt.timedelta(days=1)
    st.info(
        f"Next pending-tasks reminder: **{next_reminder.strftime('%b %d at %I:%M %p')}**"
    )

    st.divider()

    # Reset
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

    # ── 0. Inject proactive reminders from the background scheduler ──────
    # drain_reminders() is non-blocking: returns [] if nothing is queued.
    for reminder_text in drain_reminders():
        # Inject the reminder as if the user just spoke it — Ava will
        # call list_pending_tasks() and speak the summary aloud.
        agent.push_transcript(reminder_text)
        st.session_state.pending_utterance = "[⏰ Scheduled Reminder]"

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
