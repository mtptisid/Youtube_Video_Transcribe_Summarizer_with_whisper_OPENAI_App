import streamlit as st
from pytube import YouTube
import time
import os
import yt_dlp
import json
import tempfile
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.llm import LLMChain
from dotenv import load_dotenv
from google.cloud import speech
from pydub import AudioSegment

# Load environment variables
load_dotenv()

st.set_page_config(
    layout="wide"
)

# Set your Google API Key here
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/codespace/bionic-repeater-392707-e527cdea92bd.json"

# Max size for Google Speech API (10MB)
MAX_SIZE = 10 * 1024 * 1024  # 10MB

def download_video(url):
    cookie_file = '/home/codespace/cookies.txt'  # Full path to cookies file
    ydl_opts = {
        'cookiefile': cookie_file,  # Make sure this is the correct path
        'outtmpl': os.path.join('downloads/%(id)s.%(ext)s'),  # Ensure file path is valid
        'format': 'bestaudio/best',
        'verbose': True  # Optional for detailed logging
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            print(f"Downloaded: {info_dict['title']}")
            video_file_path = os.path.join('downloads', f"{info_dict['id']}.{info_dict['ext']}")
            return video_file_path
    except Exception as e:
        print(f"Error: {e}")
        return None

def transcribe_audio(file_path):
    # Initialize Google Cloud Speech client
    client = speech.SpeechClient()

    # Check if file_path is None
    if not file_path:
        return {"error": "File path is not provided."}

    # Check if the file exists before processing
    if not os.path.exists(file_path):
        return {"error": f"File not found at path: {file_path}"}

    # Convert audio file to WAV format if necessary (for compatibility)
    if file_path.endswith('.mp4') or file_path.endswith('.webm'):
        # Convert MP4 or WEBM to WAV format using pydub
        wav_path = file_path.rsplit('.', 1)[0] + ".wav"  # Replace extension with .wav
        print(f"Converting {file_path} to {wav_path}")

        try:
            audio = AudioSegment.from_file(file_path)  # pydub handles both .mp4 and .webm
            audio.export(wav_path, format="wav")
        except Exception as e:
            return {"error": f"Error during conversion: {e}"}
    else:
        wav_path = file_path  # Already a .wav file

    # Check if WAV file exists after conversion
    if not os.path.exists(wav_path):
        return {"error": f"WAV file not found at path: {wav_path}"}

    # Get the size of the audio file
    file_size = os.path.getsize(wav_path)
    if file_size > MAX_SIZE:
        # Handle large files
        return transcribe_large_audio(wav_path)

    # If file is small enough, transcribe directly
    with open(wav_path, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code="en-US",
    )

    # Perform speech recognition
    try:
        response = client.recognize(config=config, audio=audio)
    except Exception as e:
        return {"error": f"Error during speech recognition: {e}"}

    # Extract transcribed text from response
    transcribed_text = ""
    for result in response.results:
        transcribed_text += result.alternatives[0].transcript + " "

    # If no transcription was found, return an error
    if not transcribed_text:
        return {"error": "No transcription was found."}

    return {"text": transcribed_text}

def transcribe_large_audio(file_path):
    client = speech.SpeechClient()

    # Load the audio file using pydub
    audio = AudioSegment.from_file(file_path)
    
    # Convert audio to mono and set the sample rate to 16kHz
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Ensure 16-bit sample width for WAV compatibility
    audio = audio.set_sample_width(2)  # 16-bit audio (2 bytes per sample)

    # Split the audio into smaller chunks (e.g., 1 minute each)
    chunk_duration = 60 * 1000  # 1 minute in milliseconds
    chunks = []

    for start_ms in range(0, len(audio), chunk_duration):
        chunk = audio[start_ms:start_ms + chunk_duration]
        chunk_path = f"{file_path}_chunk_{start_ms}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    # Process each chunk for transcription
    full_transcript = ""
    for chunk in chunks:
        with open(chunk, 'rb') as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )

        try:
            response = client.recognize(config=config, audio=audio)
        except Exception as e:
            return {"error": f"Error during speech recognition for chunk: {e}"}

        for result in response.results:
            full_transcript += result.alternatives[0].transcript + " "

    # If no transcription was found, return an error
    if not full_transcript:
        return {"error": "No transcription was found."}

    return {"text": full_transcript}

def initialize_model():
    model = GoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    return model

def initialize_prompt_node(model):
    prompt_template = """
    Summarize the provided content in the most concise way possible.

    Content:
    {context}

    Summary:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    return LLMChain(llm=model, prompt=prompt)

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    model = initialize_model()
    prompt_node = initialize_prompt_node(model)

    context = " ".join([doc.page_content for doc in docs])

    response = prompt_node.run({"context": context})
    return response

def main():
    st.title("YouTube Video Summarizer 🎥")
    st.subheader('Built with LangChain, Gemini, Whisper, and Streamlit ❤️')

    with st.expander("About the App"):
        st.write("This app allows you to summarize YouTube videos while transcribing and processing the audio.")
        st.write("Enter a YouTube URL in the input box below and click 'Submit' to start.")

    youtube_url = st.text_input("Enter YouTube URL")

    if st.button("Submit") and youtube_url:
        start_time = time.time()

        file_path = download_video(youtube_url)
        #print(file_path)
        transcription_output = transcribe_audio(file_path)

        # Check if the transcription output contains the 'text' key
        if 'text' in transcription_output:
            text = transcription_output['text']
            text_chunks = get_text_chunks(text)
            get_vector_store(text_chunks)

            user_question = st.text_input("Ask a Question")
            if user_question:
                answer = user_input(user_question)
                st.write(answer)

            end_time = time.time()
            elapsed_time = end_time - start_time

            col1, col2 = st.columns([1, 1])

            with col1:
                st.video(youtube_url)

            with col2:
                st.header("Summarization of YouTube Video")
                st.write(f"**Transcription Time:** {elapsed_time:.2f} seconds")
                st.write(f"**Text:** {text[:500]}...")
        else:
            st.error(f"Error: {transcription_output.get('error', 'Unknown error occurred.')}")


if __name__ == "__main__":
    main()