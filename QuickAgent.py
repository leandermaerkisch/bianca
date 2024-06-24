import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model="mixtral-8x7b-32768", api_key=str(os.getenv("GROQ_API_KEY")))
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load the system prompt from a file
        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)

        start_time = time.time()

        response = self.conversation.invoke({"text": text})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['text']}")
        print(response['text'])
        return response['text']

class TextToSpeech:

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text, provider: Literal["deepgram", "elevenlabs"]):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        print(f"Sending TTS request to {provider}...")

        if provider == "deepgram":
            MODEL_NAME = "aura-asteria-en"  # Example model name, change as needed
            URL = f"https://api.beta.deepgram.com/v1/speak?model={MODEL_NAME}&performance=some&encoding=linear16&sample_rate=24000"
                    
            headers = {
                "Authorization": f"Token {os.getenv("DEEPGRAM_API_KEY")}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "text": text,
                "voice": MODEL_NAME
            }
        elif provider == "elevenlabs":
            ELEVEN_LABS_VOICE_BIANCA = "quAKExb6bIsCcLabdTA7"
            URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_LABS_VOICE_BIANCA}/stream"

            headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": str(os.getenv("ELEVENLABS_API_KEY"))
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
        if not self.is_installed(player):
            raise ValueError(f"{player} not found, necessary to stream audio.")
        
        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        with requests.post(URL, stream=True, headers=headers, json=payload) as r:
            # dg_performance_total_ms = r.headers.get('x-dg-performance-total-ms', 'Not Available')
            # print(f"Deepgram Performance Total (ms): {dg_performance_total_ms}ms")

            print(f"URL: {r.url}, Status Code: {r.status_code}, Headers: {r.headers}, Payload: {r.request.body}")

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
        

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

async def get_transcript(callback):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient("", config)

        dg_connection = deepgram.listen.asynclive.v("1")
        print ("Listening...")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            
            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                # Check if the full_sentence is not empty before printing
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="de",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=300,
            smart_format=True,
        )

        await dg_connection.start(options)

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Loop indefinitely until "goodbye" is detected
        while True:
            await get_transcript(handle_full_sentence)
            
            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                break
            
            llm_response = self.llm.process(self.transcription_response)

            tts = TextToSpeech()
            tts.speak(llm_response, provider="elevenlabs")

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())
    