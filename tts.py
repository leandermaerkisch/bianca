from deepgram import (
    DeepgramClient,
    SpeakOptions,
)
import os
from io import BytesIO
from elevenlabs.client import ElevenLabs
from elevenlabs import stream, play
from dotenv import load_dotenv
import subprocess

load_dotenv()

SPEAK_OPTIONS = {"text": "Hallo, wie kann ich Ihnen helfen?"}
filename = "your_output_file.mp3"

ELEVENLABS_API_KEY = str(os.getenv("ELEVENLABS_API_KEY"))
ELEVEN_LABS_VOICE_BIANCA = "quAKExb6bIsCcLabdTA7"

text="Hallo, wie kann ich Ihnen helfen?"


def main():
    try:
        deepgram = DeepgramClient(api_key=str(os.getenv("DEEPGRAM_API_KEY")))

        options = SpeakOptions(
            model="aura-zeus-en",
        )

        response = deepgram.speak.v("1").save(filename, SPEAK_OPTIONS, options)
        print(response.to_json(indent=4))
        player = "ffplay"

        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        player_process.stdin.write(chunk)  # type: ignore
        player_process.stdin.flush()  # type: ignore



    except Exception as e:
        print(f"Exception: {e}")

# def main():
#     try:
#         client = ElevenLabs(
#           api_key=ELEVENLABS_API_KEY
#         )
       
#         response = client.text_to_speech.convert(
#             voice_id=ELEVEN_LABS_VOICE_BIANCA,
#             optimize_streaming_latency="0",
#             output_format="mp3_22050_32",
#             text=text,
#             model_id="eleven_multilingual_v2",
#         )
#         play(response)
        
        # # print(audio_stream.to_json(indent=4))
        # audio_stream = BytesIO()

        # # Write each chunk of audio data to the stream
        # for chunk in response:
        #     if chunk:
        #         audio_stream.write(chunk)

        # # Reset stream position to the beginning
        # audio_stream.seek(0)

        # # Return the stream for further use
        # return audio_stream

    # except Exception as e:
    #     print(f"Exception: {e}")


    


if __name__ == "__main__":
    main()