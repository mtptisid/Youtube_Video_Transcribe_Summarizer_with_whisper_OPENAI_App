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