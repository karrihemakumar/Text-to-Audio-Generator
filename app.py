import streamlit as st
import torch
import numpy as np
import tempfile
import soundfile as sf
from TTS.api import TTS

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load TTS model (Tacotron2 with HiFi-GAN vocoder)
@st.cache_resource
def load_tts_model():
    return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())

# Generate audio from text
def text_to_audio(text, tts_model):
    # Synthesize audio from text
    waveform = tts_model.tts(text)
    return waveform

# Streamlit UI
st.set_page_config(page_title="Text to Audio using HiFi-GAN", layout="centered")
st.title("🗣️ Text to Audio Generator")
st.markdown("Enter a sentence and listen to it using Tacotron2 + HiFi-GAN.")

text_input = st.text_input("Enter your text here:", value="Hello, welcome to the Amazon AI Hackathon!")

if st.button("Generate Audio"):
    tts_model = load_tts_model()

    with st.spinner("Generating audio..."):
        wav = text_to_audio(text_input, tts_model)

        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            sf.write(tmpfile.name, wav, samplerate=22050)
            st.audio(tmpfile.name, format="audio/wav")
