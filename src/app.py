import re
from openai.types import model
from whisper_mic import WhisperMic
from pathlib import Path
import json
# import whisper_timestamped as whisper
import pyaudio 
from pprint import pprint
import os
import asyncio
import sys
from typing import AsyncGenerator, Generator, Optional

from openai import AsyncOpenAI
from dotenv import load_dotenv

from time import time

load_dotenv()

audio_player = pyaudio.PyAudio()

client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# Sends a prompt to the OpenAI API and returns the response as stream
# 
async def prompt_stream(prompt: str, temp: float = 0.03, max_tokens = 128, top_p: float = 1, frequency_penalty: float = 0, presence_penalty: float = 0):
    response = await client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
          temperature=temp,
          max_tokens=max_tokens,
          top_p=top_p,
          frequency_penalty=frequency_penalty,
          presence_penalty=presence_penalty,
        stream = True,
    )

    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

# Sends a prompt to the OpenAI API and returns the response as a single string
async def prompt(prompt: str, temp: float = 0.03, max_tokens = 128, top_p: float = 1, frequency_penalty: float = 0, presence_penalty: float = 0):
    start_time = time() 
    response = await client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
          temperature=temp,
          max_tokens=max_tokens,
          top_p=top_p,
          frequency_penalty=frequency_penalty,
          presence_penalty=presence_penalty,
    )
    print(f"Received all prompt results in: {int((time() - start_time) * 1000)}ms") 

    return response.choices[0].message.content

# Sends text to the OpenAI API and plays the response as voice
async def talk(input: str) -> None: 
  
    player_stream = audio_player.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True) 
  
    start_time = time() 
  
    async with client.audio.speech.with_streaming_response.create( 
        model="tts-1", 
        voice="alloy", 
        response_format="pcm",  # similar to WAV, but without a header chunk at the start. 
       input=input
    ) as response: 
        print(f"Time to first byte: {int((time() - start_time) * 1000)}ms") 
        async for chunk in response.iter_bytes(chunk_size=1024): 
            player_stream.write(chunk) 
  
    print(f"Done in {int((time() - start_time) * 1000)}ms.") 
    player_stream.close()

# Transcribes the audio from file
# Offline
# async def transcribe(file: Path) -> None:
#     audio = whisper.load_audio(file)
#
#     a = time()
#     model = whisper.load_model("tiny.en", device="cpu")
#
#     b = time()
#     result = whisper.transcribe(model, audio, language="en")
#
#     c = time()
#     print(f"Loading model: {b - a:.2f}s")
#     print(f"Transcribing: {c - b:.2f}s")
#
#     print(json.dumps(result, indent = 2, ensure_ascii = False))

# Transcribes the audio from the microphone
# Offline
async def listen_mic():
    mic = WhisperMic(model='base', english=True, implementation='faster_whisper')
    async for result in mic.listen_loop(dictate=False):
        yield result

# Breaks the text generator into a sentences generator
async def sentences_generator(text_stream: AsyncGenerator[str, None]):
    buffer = ""
    async for chunk in text_stream:
        buffer += chunk
        if sentence_match := re.search(r'([^.!?]+)([.!?])', buffer):
            sentence = sentence_match.group(1)
            punctuation = sentence_match.group(2)
            sentence = sentence + punctuation
            buffer = buffer.lstrip(sentence)
            yield sentence


async def chat_loop():
    async for result in listen_mic():
        print('You: ' + result)
        if not re.findall(r'(\[|\(|\)\])', result):
            result_generator = prompt_stream(result)
            async for sentence in sentences_generator(result_generator):
                print('Assitant: ' + sentence)
                await talk(sentence)

async def close():
    audio_player.terminate()



async def main() -> None:
    # stdin_input = sys.argv[1]
    # start = time()
    # res = await prompt(stdin_input)
    # await talk(res)
    # res = await prompt_stream(stdin_input)
    # async for chunk in prompt_stream(prompt=stdin_input):
    #     print(chunk, end='')
    # await talk(stdin_input, None)

    # print(int(time() - start))
    # await listen_mic()
    await chat_loop()

    await close()


asyncio.run(main())

# async def talk(input: str, output_path: Optional[Path]):
#     if not output_path:
#         output_path  = Path(__file__).parent / 'speech.mp3'
#
#     with_streaming = client.audio.speech.with_streaming_response
#     pprint(dir(with_streaming))
#
#     response = await with_streaming.create(
#         # TODO: Try tts-1-hd
#         model='tts-1', # or 'tts-1-hd'
#         voice='alloy',
#         response_format = 'mp3',
#         # TODO: Add speed
#         # speed
#         input=input
#     )
#
#
#     async for chunk in response:
#         pprint(dir(chunk))
#     #     if chunk.choices[0].delta.content is not None:
#     #         yield chunk.choices[0].delta.content
#     #
#     # await response.astream_to_file(output_path)
