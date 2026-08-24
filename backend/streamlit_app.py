import av
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from streamlit_webrtc import (
    AudioProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from app.sarvam_stt import SarvamRealtimeSTT, get_active_stt
from app.agent_runner import AgentRunner


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Ava — Voice AI Agent",
    page_icon="🤖",
    layout="centered",
)


st.title("🤖 Ava — Voice AI Agent")

st.caption(
    "Phase 2 — Realtime Voice Agent (Sarvam STT → Gemini → Sarvam TTS)"
)


# =========================================================
# SESSION STATE — STT & AGENT
# =========================================================

if "sarvam" not in st.session_state:
    st.session_state.sarvam = SarvamRealtimeSTT()

sarvam = st.session_state.sarvam
sarvam.bind()


if "agent" not in st.session_state:
    st.session_state.agent = AgentRunner()

agent: AgentRunner = st.session_state.agent


# =========================================================
# SESSION STATE — CONVERSATION
# =========================================================

if "chat_history" not in st.session_state:
    # Each item: {"user": str, "ava": str}
    st.session_state.chat_history = []

if "pending_utterance" not in st.session_state:
    st.session_state.pending_utterance = ""

if "partial_text" not in st.session_state:
    st.session_state.partial_text = ""

if "active_audio" not in st.session_state:
    st.session_state.active_audio = None

if "audio_turn_id" not in st.session_state:
    st.session_state.audio_turn_id = 0


# =========================================================
# AUDIO PROCESSOR  (WebRTC → STT)
# =========================================================


class AudioProcessor(AudioProcessorBase):
    """
    Runs in the WebRTC worker thread.
    Resamples browser mic audio to 16 kHz mono int16
    then hands raw bytes to SarvamRealtimeSTT.
    """

    def __init__(self):

        self.resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=16000,
        )

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:

        stt = get_active_stt()

        resampled_frames = self.resampler.resample(frame)

        for audio_frame in resampled_frames:

            audio = audio_frame.to_ndarray()

            audio = np.asarray(
                audio,
                dtype=np.int16,
            ).reshape(-1)

            volume = float(
                np.abs(audio).mean()
            )

            print(
                f"[AUDIO] "
                f"sample_rate={audio_frame.sample_rate} "
                f"shape={audio.shape} "
                f"volume={volume:.2f}"
            )

            if stt is not None:

                stt.send_audio(
                    audio.tobytes()
                )

        return frame


# =========================================================
# WEBRTC STREAMER
# =========================================================

ctx = webrtc_streamer(

    key="voice-agent",

    mode=WebRtcMode.SENDONLY,

    audio_processor_factory=AudioProcessor,

    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },

    media_stream_constraints={
        "audio": True,
        "video": False,
    },

    async_processing=True,
)


# =========================================================
# START THREADS WHEN MIC IS ACTIVE
# =========================================================

if ctx.state.playing:

    if not sarvam.running:
        sarvam.start()

    if not agent.running:
        agent.start()


# =========================================================
# UNIFIED CONVERSATION PANEL
# Single synchronized fragment to eliminate React delta collisions
# =========================================================

@st.fragment(run_every=0.3)
def conversation_panel():
    """
    Handles live transcript updates, agent answers, and seamless audio output.
    Uses a single fragment to prevent React reconciliation delta errors.
    """

    # -----------------------------------------------------
    # 1. Drain STT events
    # -----------------------------------------------------
    events = sarvam.get_events()

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

            st.error(f"Sarvam STT error: {event.get('message', 'Unknown error')}")

    # -----------------------------------------------------
    # 2. Drain Agent responses
    # -----------------------------------------------------
    new_results = agent.get_results()

    for r in new_results:

        user_text = st.session_state.pending_utterance
        st.session_state.pending_utterance = ""

        st.session_state.chat_history.append({
            "user": user_text,
            "ava": r.get("text", ""),
            "error": r.get("error", False),
        })

        # Update active audio and increment turn ID to trigger playback
        if r.get("audio_bytes") and not r.get("error"):
            st.session_state.active_audio = r["audio_bytes"]
            st.session_state.audio_turn_id += 1

    # -----------------------------------------------------
    # 3. Render Chat Dialogue
    # -----------------------------------------------------
    st.divider()

    st.subheader("💬 Live Conversation")

    has_content = (
        bool(st.session_state.chat_history)
        or bool(st.session_state.pending_utterance)
        or bool(st.session_state.partial_text)
    )

    if not has_content:
        st.caption("🎙️ Start speaking to chat with Ava...")

    # Completed conversation turns (clean text only, no old player bars)
    for turn in st.session_state.chat_history:

        if turn.get("user"):
            st.markdown(f"🗣️ **You:** {turn['user']}")

        if turn.get("ava"):
            st.markdown(f"🤖 **Ava:** {turn['ava']}")

        if turn.get("error"):
            st.warning(turn["ava"])

    # Active user utterance being formed
    if st.session_state.pending_utterance:
        st.markdown(f"🗣️ **You:** {st.session_state.pending_utterance}")

    # Active partial speech
    if st.session_state.partial_text:
        st.markdown(f"🎙️ *Listening:* {st.session_state.partial_text}")

    # -----------------------------------------------------
    # 4. Persistent Active Voice Player
    # Stays mounted across fragment reruns so audio plays in full
    # -----------------------------------------------------
    if st.session_state.active_audio:
        st.audio(
            st.session_state.active_audio,
            format="audio/wav",
            autoplay=True,
        )



# =========================================================
# RUN PANEL
# =========================================================

conversation_panel()




