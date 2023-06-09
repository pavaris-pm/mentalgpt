import numpy
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel, AutoTokenizer
from transformers import pipeline
import requests
import time


# Load the pre-trained model and tokenizer for speech to text
model_name = "wannaphong/wav2vec2-large-xlsr-53-th-cv8-deepcut"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# model for text generation - our trained mentalGPT
# mental_model = pipeline('text-generation','tontokoton/mentalgpt-gpt2')
mental_model = pipeline('text-generation','tontokoton/mentalgpt-v0.0.1')


#Define a function to transcribe audio files
def transcribe_audio(audio_file):
    # Load the audio file and resample it to the expected sampling rate
    audio_input, original_sampling_rate = torchaudio.load(audio_file)
    resampler = torchaudio.transforms.Resample(original_sampling_rate, processor.feature_extractor.sampling_rate)
    audio_input = resampler(audio_input)

    # Preprocess the audio input (normalize, resample, etc.)
    input_signal = processor(audio_input, return_tensors="pt").input_values

    # Remove the extra dimension from the input tensor
    input_signal = input_signal.squeeze(0)

    # Convert the preprocessed audio to a tensor and feed it to the model
    logits = model(input_signal).logits

    # Use the tokenizer to get the predicted transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]

    return transcription


# mentalGPT for answering mental question
def text_generation(transcription):
  output = mental_model(f"<|startoftext|>Q: {transcription}?\n\nA:  ", 
                      top_k=50, 
                      do_sample=True,
                      num_beams=5, 
                      no_repeat_ngram_size=2, 
                      early_stopping=True, 
                      max_length=150, 
                      top_p=0.95, 
                      temperature=1.9, 
                      num_return_sequences=1)
  
  return output[0]['generated_text'].split("A:")[-1]


# to convert an answer to voice forrmat (speech)
def get_speech(text): #speaker volume speed can be added as parameters
   # for full version, text need to be tokenized first
    url = "https://api-voice.botnoi.ai/api/service/generate_audio"
    payload = {"text":text, "speaker":"1", "volume":1, "speed":1, "type_media":"m4a"}
    headers = {
      # insert your botnoi token
      'Botnoi-Token': 'your token',
      'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, json=payload)
    audio_url = response.json()['audio_url']
    return audio_url


# completed function that integrates all models together
def mental_assistant(audio_file_path):

  # convert speech to text
  transcription = transcribe_audio(audio_file_path)
  print("=========== voice transcription completed ============")

  # answer question using mentalGPT
  answer_text = text_generation(transcription)
  print("=========== text generation completed ===============")

  # convert an answer into voice file (.wav)
  audio_url = get_speech(answer_text) # file will automatically saved into machine directory

  return audio_url


# Implement the model
start_time = time.time()

# input will be in format of voice file, and an output will be in the same format as well (.wav)
# do not forget to change the input!
answer_url = mental_assistant("/content/c5d85eb243a76a171f04d6360731f04a06cad7374af339a6b63db768b940d729_03152023124020153282.m4a")
print(answer_url)

end_time = time.time()
elapsed_time = end_time - start_time

print("\n")
print(f"Elapsed time: {elapsed_time} seconds.")

