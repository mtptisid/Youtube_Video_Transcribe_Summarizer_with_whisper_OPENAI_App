# YouTube Video Summarizer 🎥

## Overview

This application allows you to summarize YouTube videos by transcribing and processing their audio. Built with state-of-the-art technologies like **LangChain**, **Google Gemini**, **OpenAI Whisper**, and **Streamlit**, it provides a seamless experience to extract meaningful insights from video content.

## Features

- **YouTube Video Audio Transcription:** Converts video audio into text using OpenAI Whisper.
- **Text Summarization:** Summarizes large text chunks using LangChain and Google Gemini.
- **Question Answering:** Ask specific questions based on the summarized content.
- **Streamlined UI:** Intuitive interface built with Streamlit.

## How It Works

1. **Input a YouTube URL:** Provide the URL of the YouTube video you want to process.
2. **Audio Transcription:** The app uses OpenAI Whisper to transcribe the video audio.
3. **Summarization:** The transcribed text is summarized using LangChain and Google Gemini.
4. **Question Answering:** Interact with the summarized content to get specific answers.

## Installation

### Prerequisites

- Python 3.8 or later
- `ffmpeg` installed and available in your system's PATH

### Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2.	Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3.	Configure environment variables:
	•	Set your Google Generative AI API key in the .env file:
   ```bash
`   GOOGLE_API_KEY=your_api_key_here
   ```
   •	Set the path to your Google Cloud credentials JSON file:
   ```bash
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json
   ```
4.	Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
	1.	Start the app and input a YouTube video URL.
	2.	Wait for the app to download, transcribe, and process the video audio.
	3.	View the summarized content or ask questions in the provided input field.

## Technologies Used
	•	Streamlit: Interactive web app framework
	•	OpenAI Whisper: High-accuracy speech-to-text model
	•	LangChain: Framework for working with LLMs
	•	Google Gemini: AI-powered summarization and embeddings
	•	FAISS: Efficient similarity search and clustering

## File Structure
	•	app.py: Main application script
	•	requirements.txt: Python dependencies
	•	.env: Environment variables configuration
	•	README.md: Project documentation

## Screenshots

<img width="1440" alt="Screenshot 2024-12-15 at 1 18 35 AM" src="https://github.com/user-attachments/assets/7ad7b38f-3fbf-4a35-8795-0ef536a2d7fe" />

<img width="1440" alt="Screenshot 2024-12-15 at 1 18 41 AM" src="https://github.com/user-attachments/assets/5264e3f2-fa2a-41ec-ad00-dded7a512a0d" />

<img width="1440" alt="Screenshot 2024-12-15 at 1 18 45 AM" src="https://github.com/user-attachments/assets/2a3959d3-e8b6-4748-9964-e8b4cb09522e" />

<img width="1440" alt="Screenshot 2024-12-15 at 1 19 08 AM" src="https://github.com/user-attachments/assets/5962a426-a7b1-4f01-86af-cd06d6f49798" />

<img width="1440" alt="Screenshot 2024-12-15 at 1 20 17 AM" src="https://github.com/user-attachments/assets/ae71e6aa-b882-45b5-b06b-d1218a07ce27" />

<img width="1440" alt="Screenshot 2024-12-16 at 12 44 20 PM" src="https://github.com/user-attachments/assets/b20f00db-43bd-48cd-9ba3-b42ce989f7eb" />

<img width="1440" alt="Screenshot 2024-12-16 at 12 56 25 PM" src="https://github.com/user-attachments/assets/90f34749-ecf4-4b0f-af8e-0fbdcdb21125" />



