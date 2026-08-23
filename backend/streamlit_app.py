import av
import numpy as np
import streamlit as st

from streamlit_webrtc import (
    AudioProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from app.sarvam_stt import SarvamRealtimeSTT, get_active_stt


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Realtime Voice Agent",
    page_icon="🎙️",
    layout="centered",
)


st.title("🎙️ Realtime Voice Agent")

st.caption(
    "Phase 1 — Sarvam Realtime Speech-to-Text"
)


# =========================================================
# SARVAM SESSION
# =========================================================

if "sarvam" not in st.session_state:

    st.session_state.sarvam = (
        SarvamRealtimeSTT()
    )


sarvam = st.session_state.sarvam
sarvam.bind()


# =========================================================
# TRANSCRIPT STATE
# =========================================================

if "final_text" not in st.session_state:

    st.session_state.final_text = []


if "partial_text" not in st.session_state:

    st.session_state.partial_text = ""


# =========================================================
# AUDIO PROCESSOR
# =========================================================

class AudioProcessor(AudioProcessorBase):

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
# WEBRTC
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
# START SARVAM ONLY WHEN MIC IS ACTIVE
# =========================================================

if ctx.state.playing:

    if not sarvam.running:

        sarvam.start()


# =========================================================
# TRANSCRIPT PANEL
# =========================================================

@st.fragment(run_every=0.2)
def transcript_panel():

    events = sarvam.get_events()

    # -----------------------------------------------------
    # Process Sarvam events
    # -----------------------------------------------------

    for event in events:

        event_type = event.get("type")

        # ---------------------------------------------
        # Partial
        # ---------------------------------------------

        if event_type == "partial":

            st.session_state.partial_text = (
                event.get("text", "")
            )

        # ---------------------------------------------
        # Final
        # ---------------------------------------------

        elif event_type == "final":

            text = (
                event.get("text", "")
                .strip()
            )

            if text:

                st.session_state.final_text.append(
                    text
                )

            st.session_state.partial_text = ""

        # ---------------------------------------------
        # Error
        # ---------------------------------------------

        elif event_type == "error":

            st.error(
                f"Sarvam error: "
                f"{event.get('message', 'Unknown error')}"
            )

    # -----------------------------------------------------
    # UI
    # -----------------------------------------------------

    st.divider()

    st.subheader("Live Transcript")

    if not st.session_state.final_text:

        if not st.session_state.partial_text:

            st.caption(
                "🎙️ Start speaking..."
            )

    # Final transcripts

    for text in st.session_state.final_text:

        st.markdown(
            f"🗣️ **You:** {text}"
        )

    # Current partial transcript

    if st.session_state.partial_text:

        st.markdown(
            f"🎙️ **Listening:** "
            f"{st.session_state.partial_text}"
        )


# =========================================================
# RUN TRANSCRIPT PANEL
# =========================================================

transcript_panel()