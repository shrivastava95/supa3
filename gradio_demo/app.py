import gradio as gr
import torch
from torchvision.transforms import ToTensor
import os, sys
os.chdir('../SSL_Anti-Spoofing')
sys.path.append(os.path.abspath('../SSL_Anti-Spoofing'))

from gradio_dump import model, pad, loader_rishabhsubset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()

# Define the function to process the audio files
def process_audio(audio_file):
    audio = loader_rishabhsubset(audio_file)
    batch = [audio]
    with torch.no_grad():
        inputs = torch.stack(batch).to(device)
        scores = model(inputs)[:, 1]
        scores = scores.data.cpu().numpy().tolist()
    threshold = -3.45
    labels = ['Spoof!' if score > threshold else 'Real!' for score in scores]
    return labels[0]

def preprocess_audio(audio_file):
    audio_tensor = ToTensor()(audio_file)
    return audio_tensor

inputs = gr.inputs.Audio(source='upload', label="Upload audio files", type="filepath")
outputs = gr.outputs.Label(label="Predicted Labels - 0 is Spoof, 1 is Real")
gr.Interface(fn=process_audio, inputs=[inputs], outputs=outputs).launch(share=True)