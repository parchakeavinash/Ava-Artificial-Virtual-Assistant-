import base64
import datetime as dt
import logging
import threading
import uuid
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
# PAGE CONFIG (ChatGPT-Style Wide Layout)
# =========================================================
st.set_page_config(
    page_title="Ava - Voice & Chat AI Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for ChatGPT-like styling
st.markdown(
    """
    <style>
    /* Sleek chat message styling */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }
    /* Sticky bottom input area */
    .main .block-container {
        padding-bottom: 80px;
    }
    /* Session button list styling */
    .sidebar-session-btn {
        text-align: left !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
REMINDER_HOUR = 9
REMINDER_MINUTE = 0

# Module-level references for WebRTC background audio thread
_ACTIVE_STT_CLIENT: SarvamRealtimeSTT | None = None
_AUDIO_RESAMPLER: av.AudioResampler | None = None

if "stt_client" not in st.session_state:
    st.session_state.stt_client = SarvamRealtimeSTT()
    st.session_state.stt_client.start()

_ACTIVE_STT_CLIENT = st.session_state.stt_client

if "whisper_stt" not in st.session_state:
    st.session_state.whisper_stt = GroqWhisperSTT()

if "agent_runner" not in st.session_state:
    st.session_state.agent_runner = AgentRunner()
    st.session_state.agent_runner.start()

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = f"chat_{uuid.uuid4().hex[:8]}"

# Ensure agent runner is pointing to current session
st.session_state.agent_runner.set_session_id(st.session_state.current_session_id)

# Background reminder scheduler
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


def load_session_history(session_id: str):
    """Loads database message history for the selected session."""
    raw_msgs = st.session_state.agent_runner.agent.memory.short_term.get_raw_messages(
        session_id=session_id
    )
    history = []
    # Pair messages into user / assistant turns
    i = 0
    while i < len(raw_msgs):
        msg = raw_msgs[i]
        if msg.role == "user":
            user_text = msg.content
            ava_text = ""
            if i + 1 < len(raw_msgs) and raw_msgs[i + 1].role == "assistant":
                ava_text = raw_msgs[i + 1].content
                i += 1
            history.append({"user": user_text, "ava": ava_text, "error": False})
        elif msg.role == "assistant":
            history.append({"user": "", "ava": msg.content, "error": False})
        i += 1
    return history


# =========================================================
# AUDIO PROCESSOR CALLBACK FOR WEBRTC
# =========================================================
def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    """Resample to 16kHz mono linear16 and stream to STT without session_state context."""
    global _ACTIVE_STT_CLIENT, _AUDIO_RESAMPLER
    try:
        if _ACTIVE_STT_CLIENT is not None and _ACTIVE_STT_CLIENT.running:
            if _AUDIO_RESAMPLER is None:
                _AUDIO_RESAMPLER = av.AudioResampler(format="s16", layout="mono", rate=16000)
            resampled_frames = _AUDIO_RESAMPLER.resample(frame)
            for rf in resampled_frames:
                pcm_bytes = rf.to_ndarray().tobytes()
                _ACTIVE_STT_CLIENT.push_audio(pcm_bytes)
    except Exception as exc:
        logging.debug(f"[WebRTC] Audio callback error: {exc}")
    return frame


# =========================================================
# LEFT SIDEBAR: CHATGPT-STYLE SESSIONS & MEMORY
# =========================================================
with st.sidebar:
    st.markdown("### 🎙️ **Ava Assistant**")

    # "+ New Chat" Button
    if st.button("➕ **New Chat**", use_container_width=True, type="primary"):
        old_sid = st.session_state.current_session_id
        # Background auto-distillation of previous session
        threading.Thread(
            target=lambda: st.session_state.agent_runner.agent.auto_distill_if_needed(session_id=old_sid),
            daemon=True,
        ).start()

        new_sid = f"chat_{uuid.uuid4().hex[:8]}"
        st.session_state.current_session_id = new_sid
        st.session_state.agent_runner.set_session_id(new_sid)
        st.session_state.chat_history = []
        st.session_state.active_audio = None
        st.session_state.partial_text = ""
        st.session_state.pending_utterance = ""
        st.rerun()

    st.markdown("---")
    st.markdown("##### 💬 **Recent Conversations**")

    # Retrieve all saved sessions from DB
    sessions = st.session_state.agent_runner.agent.list_sessions()

    # Ensure current session appears in list
    existing_ids = [s["session_id"] for s in sessions]
    if st.session_state.current_session_id not in existing_ids:
        sessions.insert(0, {
            "session_id": st.session_state.current_session_id,
            "last_message": "New conversation...",
            "last_active": dt.datetime.now(),
            "message_count": 0,
        })

    for s in sessions:
        sid = s["session_id"]
        is_active = (sid == st.session_state.current_session_id)
        label_prefix = "🟢 " if is_active else "💬 "
        preview = s.get("last_message", "Chat") or "Chat"
        if len(preview) > 26:
            preview = preview[:26] + "..."

        btn_label = f"{label_prefix}{preview}"
        if st.button(btn_label, key=f"session_btn_{sid}", use_container_width=True):
            if sid != st.session_state.current_session_id:
                old_sid = st.session_state.current_session_id
                threading.Thread(
                    target=lambda: st.session_state.agent_runner.agent.auto_distill_if_needed(session_id=old_sid),
                    daemon=True,
                ).start()

                st.session_state.current_session_id = sid
                st.session_state.agent_runner.set_session_id(sid)
                st.session_state.chat_history = load_session_history(sid)
                st.session_state.active_audio = None
                st.session_state.partial_text = ""
                st.session_state.pending_utterance = ""
                st.rerun()

    st.markdown("---")

    # 1. Semantic Facts (What Ava Knows About You)
    with st.expander("🧠 **What Ava Knows (Memory)**", expanded=False):
        facts = st.session_state.agent_runner.agent.list_known_facts()
        if facts:
            for f in facts:
                st.markdown(f"• **{f['key'].replace('_', ' ').title()}:** {f['value']}")
        else:
            st.caption("No personal facts saved yet. Chat naturally and Ava will remember your preferences!")

        if st.button("Distill Session Episode 🏛️", use_container_width=True):
            ep = st.session_state.agent_runner.agent.create_episode(
                session_id=st.session_state.current_session_id
            )
            if ep:
                st.success(f"Saved episode: {ep['summary']}")
            else:
                st.info("Insufficient conversation to distill an episode.")

    # 2. System Status & Quotas
    with st.expander("⚙️ **Engine & Status**", expanded=False):
        metrics = agent_metrics.get_stats()
        st.markdown(f"**LLM:** `{metrics['last_provider']}`")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Groq", metrics["groq_calls"])
        with col2:
            st.metric("Gemini", metrics["gemini_fallback_calls"])

        w_stats = whisper_tracker.get_stats()
        st.caption(f"Whisper STT: {w_stats['active_model']} ({w_stats['rpm_remaining']}/{w_stats['rpm_limit']} RPM)")

        # Proactive reminder status
        next_reminder = dt.datetime.now().replace(
            hour=REMINDER_HOUR, minute=REMINDER_MINUTE, second=0, microsecond=0
        )
        if next_reminder < dt.datetime.now():
            next_reminder = next_reminder + dt.timedelta(days=1)
        st.caption(f"Next Reminder: {next_reminder.strftime('%b %d at %I:%M %p')}")

        if st.button("🧹 Clear This Session", use_container_width=True):
            st.session_state.agent_runner.reset_conversation(session_id=st.session_state.current_session_id)
            st.session_state.chat_history.clear()
            st.session_state.active_audio = None
            st.rerun()


# =========================================================
# MAIN AREA: CHAT INTERFACE WITH DUAL TEXT + VOICE
# =========================================================

# Top Header Bar
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown(f"### 💬 Session: `{st.session_state.current_session_id}`")
with header_col2:
    st.caption("🚀 Model: **Groq (`openai/gpt-oss-120b`)**")

# Voice Mode Accordion / Bar
with st.expander("🎙️ **Voice Mode (Live Streaming Mic & Quick Voice Note)**", expanded=True):
    v_tab1, v_tab2 = st.tabs(["🔴 Live Streaming Mic (WebRTC)", "🎙️ Quick Voice Note"])

    with v_tab1:
        v_col1, v_col2 = st.columns([2, 3])
        with v_col1:
            webrtc_streamer(
                key="ava-mic",
                mode=WebRtcMode.SENDONLY,
                audio_frame_callback=audio_frame_callback,
                media_stream_constraints={"video": False, "audio": True},
            )
        with v_col2:
            if st.session_state.partial_text:
                st.markdown(f"🎙️ *Listening:* `{st.session_state.partial_text}`")
            elif getattr(st.session_state.agent_runner, "is_busy", False):
                st.info(f"🔎 {getattr(st.session_state.agent_runner, 'current_action', 'Ava is thinking & using tools...')}")
            else:
                st.caption("Click **START** to speak continuously with Ava in real-time.")

    with v_tab2:
        quick_audio = st.audio_input("Record a voice message for Ava:")
        if quick_audio is not None:
            audio_id = f"{quick_audio.name}_{quick_audio.size}"
            if st.session_state.get("last_quick_audio_id") != audio_id:
                st.session_state.last_quick_audio_id = audio_id
                with st.spinner("Transcribing your voice..."):
                    try:
                        raw_bytes = quick_audio.read()
                        transcript = st.session_state.whisper_stt.client.audio.transcriptions.create(
                            file=("audio.wav", raw_bytes),
                            model=settings.GROQ_WHISPER_PRIMARY,
                            temperature=0.0,
                            response_format="json",
                        ).text.strip()
                    except Exception as stt_err:
                        st.error(f"Transcription error: {stt_err}")
                        transcript = ""

                if transcript:
                    st.session_state.chat_history.append({"user": transcript, "ava": "", "error": False})
                    with st.spinner("Ava is responding..."):
                        reply = st.session_state.agent_runner.agent.respond(
                            user_text=transcript,
                            session_id=st.session_state.current_session_id,
                        )
                        st.session_state.chat_history[-1]["ava"] = reply
                        try:
                            reply_audio = st.session_state.agent_runner.tts.speak(reply)
                            st.session_state.active_audio = reply_audio
                        except Exception as tts_err:
                            logging.warning(f"TTS error: {tts_err}")
                            st.session_state.active_audio = None
                    st.rerun()

st.markdown("---")


# =========================================================
# LIVE CONVERSATION STREAM (Auto-refresh fragment for Voice)
# =========================================================
@st.fragment(run_every=0.4)
def live_chat_stream():
    stt = st.session_state.stt_client
    agent = st.session_state.agent_runner

    # 1. Drain proactive reminders
    for reminder_text in drain_reminders():
        agent.push_transcript(reminder_text)
        st.session_state.pending_utterance = "[⏰ Scheduled Reminder]"

    # 2. Drain STT events
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

    # 3. Drain completed voice results from AgentRunner
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

    # 4. Render Conversation Dialogue in ChatGPT bubbles
    # If session is empty, load existing messages from DB
    if not st.session_state.chat_history:
        db_history = load_session_history(st.session_state.current_session_id)
        if db_history:
            st.session_state.chat_history = db_history

    if not st.session_state.chat_history and not st.session_state.pending_utterance:
        st.info("👋 Hello! I am Ava. You can **speak** using the microphone above, or **type** a message below.")

    for turn in st.session_state.chat_history:
        if turn.get("user"):
            with st.chat_message("user", avatar="👤"):
                st.markdown(turn["user"])
        if turn.get("ava"):
            with st.chat_message("assistant", avatar="🎙️"):
                st.markdown(turn["ava"])
                if turn.get("error"):
                    st.warning("⚠️ Error encountered during response.")

    if st.session_state.pending_utterance:
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"*{st.session_state.pending_utterance}*")

    # Audio playback for voice replies
    if st.session_state.active_audio:
        st.audio(
            st.session_state.active_audio,
            format="audio/wav",
            autoplay=True,
        )


# Render conversation panel
live_chat_stream()


# =========================================================
# TEXT INPUT (ChatGPT-Style Bottom Chat Bar)
# =========================================================
user_input = st.chat_input("Message Ava or ask a question...")

if user_input:
    # 1. Add user message immediately
    st.session_state.chat_history.append({
        "user": user_input,
        "ava": "",
        "error": False,
    })

    with st.spinner("Ava is thinking..."):
        # 2. Invoke resilient agent with active session ID
        response_text = st.session_state.agent_runner.agent.respond(
            user_text=user_input,
            session_id=st.session_state.current_session_id,
        )

        # 3. Synthesize speech via Sarvam TTS
        try:
            audio_bytes = st.session_state.agent_runner.tts.speak(response_text)
            st.session_state.active_audio = audio_bytes
        except Exception as tts_err:
            logging.warning(f"TTS error: {tts_err}")
            st.session_state.active_audio = None

        # 4. Update the turn
        if st.session_state.chat_history:
            st.session_state.chat_history[-1]["ava"] = response_text

    st.rerun()
