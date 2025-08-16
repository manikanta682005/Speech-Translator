import os
import uuid
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from gtts import gTTS
import pygame
import speech_recognition as sr
from googletrans import Translator, LANGUAGES
from langdetect import detect
import torch
import torch.nn as nn
import numpy as np
 
# Initialize translator and pygame for audio playback
translator = Translator()
pygame.mixer.init()

# Custom Transformer-based model and tokenizer
class MultiHeadAttention(nn.Module):
    def init(self, d_model, num_heads):
        super().init()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V):
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

class SimpleTransformer(nn.Module):
    def init(self, vocab_size=10000, d_model=256, num_classes=2):
        super().init()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = MultiHeadAttention(d_model, num_heads=8)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.attention(x, x, x)
        x = x.mean(dim=1)
        return torch.softmax(self.fc(x), dim=1)

model = SimpleTransformer()
model.eval()

def classify_language(text):
    try:
        return detect(text)
    except:
        return "Unknown"

def translate_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        messagebox.showinfo("Info", "Speak now...")
        recognizer.pause_threshold = 1
        audio = recognizer.listen(source)
    try:
        input_text = recognizer.recognize_google(audio)
        detected_lang = classify_language(input_text)
        detected_lang_name = LANGUAGES.get(detected_lang, 'Unknown')
        detected_lang_label.config(text=f"Detected Language: {detected_lang_name}")
        
        to_lang_code = language_var.get().split(' - ')[0]
        translated_text = translator.translate(input_text, dest=to_lang_code).text
        translated_text_display.config(state="normal")
        translated_text_display.delete("1.0", tk.END)
        translated_text_display.insert(tk.END, translated_text)
        translated_text_display.config(state="disabled")
        
        # Play the translated text as voice
        text_to_voice(translated_text, to_lang_code)
    except sr.UnknownValueError:
        messagebox.showerror("Error", "Could not understand the audio.")
    except sr.RequestError:
        messagebox.showerror("Error", "Could not request results; check your network.")

def text_to_voice(text, lang_code):
    filename = f"{uuid.uuid4()}.mp3"
    tts = gTTS(text=text, lang=lang_code, slow=False)
    tts.save(filename)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.unload()
    os.remove(filename)

# Initialize GUI with Tkinter
root = tk.Tk()
root.title("Voice-Only Translator & Text Analyzer")
root.geometry("600x400")

# Language selection dropdown
language_var = tk.StringVar(root)
language_dropdown = ttk.Combobox(root, textvariable=language_var, state="readonly", width=40)
language_dropdown['values'] = [f"{code} - {name}" for code, name in LANGUAGES.items()]
language_dropdown.set("en - English")
language_dropdown.pack(pady=10)

# Detected language label
detected_lang_label = tk.Label(root, text="Detected Language: ")
detected_lang_label.pack(pady=5)

# Translate voice button
voice_button = tk.Button(root, text="Translate Voice", command=translate_voice)
voice_button.pack(pady=10)

# Translated text display
translated_text_label = tk.Label(root, text="Translated Text:")
translated_text_label.pack(pady=5)
translated_text_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=10, state="disabled")
translated_text_display.pack(pady=10)

root.mainloop()