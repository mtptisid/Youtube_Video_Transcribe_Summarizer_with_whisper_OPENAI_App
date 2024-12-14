import os
import subprocess
import streamlit as st
from pytube import YouTube
import time
import yt_dlp
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(layout="wide")

# Function to re-encode audio file to opus (ogg) format
def reencode_audio_to_ogg(input_file, output_file="encoded_audio.ogg"):
    command = [
        "ffmpeg", "-y",
        "-i", input_file, "-vn", "-map_metadata", "-1",
        "-ac", "1", "-c:a", "libopus", "-b:a", "12k", "-application", "voip", output_file
    ]
    subprocess.run(command, check=True)
    return output_file

# Function to download YouTube video
def download_video(url):
    cookie_file = '/home/codespace/cookies.txt'
    ydl_opts = {
        'cookiefile': cookie_file,
        'outtmpl': os.path.join('downloads/%(id)s.%(ext)s'),
        'format': 'bestaudio/best',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_file_path = os.path.join('downloads', f"{info_dict['id']}.{info_dict['ext']}")
            return video_file_path
    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return None

# Function to transcribe audio using Whisper
def transcribe_audio_with_whisper(file_path, model_name="openai/whisper-medium", target_language="en"):
    try:
        # Load Whisper model and processor
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)

        # Load and preprocess audio
        audio, rate = librosa.load(file_path, sr=16000, mono=True)
        inputs = processor(audio, sampling_rate=rate, return_tensors="pt", padding=True)

        # Force the model to transcribe in the target language
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=target_language)

        # Generate transcription
        with torch.no_grad():
            generated_ids = model.generate(inputs.input_features, forced_decoder_ids=forced_decoder_ids)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return {"text": transcription}
    except Exception as e:
        return {"error": f"Error during transcription: {e}"}

# Streamlit App
def main():
    st.title("YouTube Video Summarizer 🎥")
    st.subheader('Built with LangChain, Whisper, and Streamlit ❤️')

    with st.expander("About the App"):
        st.write("This app allows you to summarize YouTube videos by transcribing and processing the audio.")
        st.write("Enter a YouTube URL in the input box below and click 'Submit' to start.")

    youtube_url = st.text_input("Enter YouTube URL")

    if st.button("Submit") and youtube_url:
        start_time = time.time()

        st.info("Downloading video...")
        file_path = download_video(youtube_url)
        if not file_path:
            st.error("Failed to download video.")
            return

        # Re-encode audio to OGG format
        st.info("Re-encoding audio...")
        try:
            ogg_file_path = reencode_audio_to_ogg(file_path)
        except Exception as e:
            st.error(f"Audio re-encoding failed: {e}")
            return

        # Transcribe audio
        st.info("Transcribing audio...")
        transcription_output = transcribe_audio_with_whisper(ogg_file_path)

        # Check transcription output
        if 'text' in transcription_output:
            text = transcription_output['text']
            st.success("Transcription completed!")
            st.write(f"**Transcription:** {text[:500]}...")

            # Placeholder: Add your summarization/QA pipeline here
            st.write("**Note:** Integrate your summarization or QA pipeline for further processing.")
        else:
            st.error(f"Error: {transcription_output.get('error', 'Unknown error occurred.')}")

        end_time = time.time()
        elapsed_time = end_time - start_time

        st.write(f"**Processing Time:** {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()