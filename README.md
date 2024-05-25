# Bianca

AI for medical doctors

## Installation

This is a alpha demo showing a bot that uses Text-To-Speech, Speech-To-Text, and a language model to have a conversation with a user.

This demo is set up to use Deepgram for the audio service and Groq the LLM.

This demo utilizes streaming for sst and tts to speed things up.

The files in building_blocks are the isolated components if you'd like to inspect them

`python3 QuickAgent.py`

Install portaudio on your system. On macOS, you can do this with the following command:

`brew install portaudio`

Then, install the required Python packages:
`uv pip install -r requirements.txt`
