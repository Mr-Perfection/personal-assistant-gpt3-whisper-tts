#!/usr/bin/env python
# coding: utf-8

#hide
# setup 
# https://github.com/openai/whisper
# pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git


#hide


#hide


import whisper

model = whisper.load_model("base")


#hide


#hide
import sounddevice as sd
from scipy.io.wavfile import write

# Sampling frequency
# Regardless of the sampling rate used in the original audio file, 
# the audio signal gets resampled to 16kHz (via ffmpeg). Anything grater than 16kHz should work.
# see https://github.com/openai/whisper/discussions/870.
freq = 44100

# Recording duration in seconds
# duration = int(input("select duration of the audio: "))


questions_path = "./questions"


# #hide
# # Start recorder with the given values of 
# # duration and sample frequency.

# recording = sd.rec(int(duration * freq), 
#                    samplerate=freq, channels=2)

# # Record audio for the given number of seconds
# sd.wait()


# #hide
# write("question1.wav", freq, recording)


audio_file = input("what's your audio called? ")
audio_path = f"{questions_path}/{audio_file}"


result = model.transcribe(audio_path)


result = whisper.transcribe(model, audio_path, language='en', fp16=False)
result


#hide


import os
import openai

from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


openai.Model.list()


prompt = result['text'].strip()
prompt


# API reference: https://platform.openai.com/docs/api-reference/completions/create
response = openai.Completion.create(
  model="text-davinci-003",
  prompt=prompt,
  max_tokens=1000,
  temperature=0.6
)


response


#hide


from TTS.api import TTS
# https://tts.readthedocs.io/en/latest/inference.html
TTS.list_models()


tts = TTS('tts_models/multilingual/multi-dataset/your_tts')


text = response["choices"][0]["text"]
tts.tts_to_file(text=text, speaker=tts.speakers[0], language=tts.languages[0], file_path=f"{questions_path}/output.wav")






