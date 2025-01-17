import os
import requests
from dotenv import load_dotenv
import subprocess
import shutil
from typing import Literal
import time
from deepgram import Deepgram

# Load environment variables
load_dotenv()

# Set your Deepgram API Key and desired voice model
DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
MODEL_NAME = "alpha-stella-en"  # Example model name, change as needed

ELEVENLABS_API_KEY = str(os.getenv("ELEVENLABS_API_KEY"))

def is_installed(lib_name: str) -> bool:
    lib = shutil.which(lib_name)
    return lib is not None

def play_stream(audio_stream, use_ffmpeg=True):
    player = "ffplay"
    if not is_installed(player):
        raise ValueError(f"{player} not found, necessary to stream audio.")
    
    player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
    player_process = subprocess.Popen(
        player_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for chunk in audio_stream:
        if chunk:
            player_process.stdin.write(chunk)  # type: ignore
            player_process.stdin.flush()  # type: ignore
    
    if player_process.stdin:
        player_process.stdin.close()
    player_process.wait()

def send_tts_request(text: str, provider: Literal["deepgram", "elevenlabs"]):
    print(f"Sending TTS request to {provider}...")

    if provider == "deepgram":
        URL = f"https://api.beta.deepgram.com/v1/speak?model={MODEL_NAME}&performance=some&encoding=linear16&sample_rate=24000"
                
        headers = {
            "Authorization": f"Token {DG_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": text,
            "voice": MODEL_NAME
        }
    elif provider == "elevenlabs":
        ELEVEN_LABS_VOICE_BIANCA = "quAKExb6bIsCcLabdTA7"
        URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_LABS_VOICE_BIANCA}" # /stream

        headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
        }

        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "optimize_streaming_latency": "0",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "use_speaker_boost": True
            },
        }

    
    start_time = time.time()  # Record the time before sending the request
    first_byte_time = None  # Initialize a variable to store the time when the first byte is received
    
    # Initialize the player process here, before receiving the stream
    player = "ffplay"
    if not is_installed(player):
        raise ValueError(f"{player} not found, necessary to stream audio.")
    
    player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
    player_process = subprocess.Popen(
        player_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    start_time = time.time()  # Record the time before sending the request
    first_byte_time = None  # Initialize a variable to store the time when the first byte is received
    
    with requests.post(URL, stream=True, headers=headers, json=payload) as r:
        # dg_performance_total_ms = r.headers.get('x-dg-performance-total-ms', 'Not Available')
        # print(f"Deepgram Performance Total (ms): {dg_performance_total_ms}ms")

        print(payload)

        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                if first_byte_time is None:  # Check if this is the first chunk received
                    first_byte_time = time.time()  # Record the time when the first byte is received
                    ttfb = int((first_byte_time - start_time)*1000)  # Calculate the time to first byte
                    print(f"Time to First Byte (TTFB): {ttfb}ms")
                # Write each chunk to the player's stdin immediately
                player_process.stdin.write(chunk)  # type: ignore
                player_process.stdin.flush()  # type: ignore

    # Close the player's stdin and wait for the process to finish
    if player_process.stdin:
        player_process.stdin.close()
    player_process.wait()

# Example usage with saving to file
text = """
The returns for performance are superlinear."""
send_tts_request(text, provider="elevenlabs")