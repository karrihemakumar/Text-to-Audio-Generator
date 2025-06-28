# Text-to-Audio-Generator
Enter a sentence and listen to it using Tacotron2 + HiFi-GAN


🎯 Purpose / Objective
This application enables users to enter text in a simple web interface and instantly hear it spoken aloud using a neural text-to-speech (TTS) pipeline. It is designed to:

Demonstrate the power of neural TTS models (Tacotron2 + HiFi-GAN)

Allow real-time text-to-audio synthesis via a lightweight and deployable frontend

Run efficiently even on consumer-grade hardware or CPU

🔬 Methodology
1. Libraries and Dependencies
Streamlit: Provides a reactive web interface.

TTS (coqui-ai): Simplifies access to pre-trained text-to-speech models.

Torch: Handles backend GPU acceleration and deep learning model execution.

Soundfile: Saves generated waveform as an audio file.

Tempfile: Creates temporary .wav files for playback.

2. Model Used
Tacotron2-DDC (tts_models/en/ljspeech/tacotron2-DDC): Converts text to mel spectrograms.

HiFi-GAN (used automatically by the TTS API): Converts mel spectrograms to waveforms (audio).

These models are both pre-trained on the LJ Speech Dataset and are well-suited for high-quality English voice synthesis.

3. Streamlit App Workflow
a. Device Selection
python
Copy
Edit
device = "cuda" if torch.cuda.is_available() else "cpu"
Automatically selects GPU if available; otherwise, uses CPU.

b. Model Loader
python
Copy
Edit
@st.cache_resource
def load_tts_model():
    return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())
Loads and caches the TTS model to avoid reloading on every run.

c. Text Input
python
Copy
Edit
text_input = st.text_input(...)
Provides a text box for the user to input the sentence to be spoken.

d. TTS Execution
python
Copy
Edit
wav = text_to_audio(text_input, tts_model)
Passes the input text to the model, which returns a waveform.

e. WAV File Generation
python
Copy
Edit
sf.write(tmpfile.name, wav, samplerate=22050)
Writes the waveform to a temporary .wav file using the standard 22.05 kHz sampling rate.

f. Audio Playback
python
Copy
Edit
st.audio(tmpfile.name, format="audio/wav")
Streamlit renders an HTML audio player that plays the generated voice.

🧪 Output / Results
The system outputs natural-sounding English speech from the typed input.

It runs entirely inside a browser with no dependencies beyond Python, making it ideal for deployment on platforms like Streamlit Cloud, Hugging Face Spaces, or Docker.

No fine-tuning or training is needed — the app uses off-the-shelf models.

🧰 How to Run
✅ Installation
bash
Copy
Edit
pip install streamlit torch TTS soundfile
▶️ Run the App
bash
Copy
Edit
streamlit run app.py
📌 Key Advantages
Lightweight and deployable

Real-time voice generation

Compatible with CPU or GPU

Extensible: supports multilingual or multi-speaker models

💡 Possible Extensions
Support for multiple voices / accents

Download button for saving audio

Integration with character animation or video dubbing

Batch processing for long-form narration
