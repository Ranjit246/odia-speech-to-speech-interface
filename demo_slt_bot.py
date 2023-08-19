### Import Dependencies
import tempfile
from ttsmms import TTS
import soundfile as sf
import io
import numpy as np
from fastapi.responses import FileResponse
import librosa
import shutil
import torch
from transformers import Wav2Vec2ForCTC, AutoProcessor, AutoTokenizer, AutoModelForCausalLM

################### STT Part ######################

model_id = "facebook/mms-1b-all"
processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)
# Add language adapters for Odia ('ory')
processor.tokenizer.set_target_lang("ory")
model.load_adapter("ory")

audio = "input audio file path"

def transcribe(audio):
    y, _ = librosa.load(audio)
    inputs = processor(y, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).logits
    ids = torch.argmax(outputs, dim=-1)[0]
    text = processor.decode(ids)
    return  text

############# LLM Part ###############

text1 = transcribe(audio)

# Specify device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer 
tokenizer = AutoTokenizer.from_pretrained("Ranjit/llama_v2_or")
model = AutoModelForCausalLM.from_pretrained("Ranjit/llama_v2_or", trust_remote_code=True, torch_dtype=torch.float16).to(device)

# Tokenize input 
inputs = tokenizer(text1, return_tensors="pt").to(device)

# Generate output
with torch.no_grad():
  outputs = model.generate(input_ids=inputs["input_ids"], 
                           attention_mask=inputs["attention_mask"],
                           max_new_tokens=1024, 
                           pad_token_id=tokenizer.eos_token_id)

text2 = tokenizer.decode(outputs[0], skip_special_tokens=True)
################## TTS Part #####################

tts = TTS("ory")

def generate_audio(text):
    wav = tts.synthesis(text)
    audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_path = audio_file.name
    sf.write(audio_path, wav["x"], wav["sampling_rate"])
    return audio_path

a = generate_audio(text2)
destination_file = "file path give here"
shutil.copy(a, destination_file)
