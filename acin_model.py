# app.py
# Affective Contextual Analysis Framework (ACAF) - v2.5 (Dynamic Video Generation)
# Features:
# - NEW: Generates dynamic, animated videos for each emotion instead of static colors.
# - Fixed RuntimeError by padding/truncating audio to a fixed length.
# - Fixed moviepy error by using older syntax for audio attachment.
# - Made classification_report more robust to prevent crashes on incomplete data.
# - Corrected 'sampling_rate' attribute for Wav2Vec2Processor.
# - Swapped TimeSformer for Vision Transformer (ViT).

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    XLMRobertaModel, XLMRobertaTokenizer,
    Wav2Vec2Model, Wav2Vec2Processor,
    ViTModel, AutoImageProcessor
)
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import warnings
import io
import re
from gtts import gTTS
from moviepy import ImageSequenceClip, AudioFileClip
import shutil
from sklearn.metrics import f1_score, classification_report
import plotly.express as px
import tempfile
import math

# --- OLLAMA CHECK ---
try:
    import ollama
except ImportError:
    st.error("The 'ollama' library is not installed. Please run 'pip install ollama' to use the synthetic data generation feature.")
    st.stop()

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
class Config:
    SYNTHETIC_DATA_DIR = 'synthetic_acaf_data'
    MODEL_SAVE_PATH = 'acaf_model.pth'
    OLLAMA_MODEL = 'gemma:2b'
    
    NUM_EMOTIONS = 7
    EMOTION_MAP = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}
    ID_TO_EMOTION = {v: k for k, v in EMOTION_MAP.items()}
    EMOTION_COLORS = { # BGR colors for OpenCV
        'anger': (0, 0, 255), 'disgust': (0, 128, 0), 'fear': (75, 0, 130),
        'joy': (0, 255, 255), 'neutral': (128, 128, 128), 'sadness': (255, 0, 0), 'surprise': (0, 165, 255)
    }
    
    TEXT_MODEL = 'xlm-roberta-base'
    AUDIO_MODEL = 'facebook/wav2vec2-base-960h'
    VIDEO_MODEL = 'google/vit-base-patch16-224-in21k'
    EMBED_DIM = 768
    FUSION_DIM = 256
    
    EPOCHS = 5
    BATCH_SIZE = 2
    LEARNING_RATE = 5e-5
    MAX_LEN = 128
    AUDIO_MAX_LENGTH = 80000 # Fixed length for audio (5 seconds at 16kHz)

# --- 2. DYNAMIC VIDEO FRAME GENERATION ---
def generate_emotion_frame(emotion, frame_index, total_frames=16):
    """Generates a single video frame with dynamic patterns based on emotion."""
    height, width = 224, 224
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    color = Config.EMOTION_COLORS[emotion]
    progress = frame_index / total_frames

    if emotion == 'joy':
        radius = int(progress * 100)
        cv2.circle(frame, (width // 2, height // 2), radius, color, -1)
    elif emotion == 'anger':
        for i in range(5):
            x1 = np.random.randint(0, width)
            y1 = np.random.randint(0, height)
            x2 = np.random.randint(0, width)
            y2 = np.random.randint(0, height)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
    elif emotion == 'sadness':
        y_pos = int(progress * height)
        cv2.line(frame, (width // 2, 0), (width // 2, y_pos), color, 10)
    elif emotion == 'surprise':
        thickness = int((1 - progress) * 20) + 1
        cv2.circle(frame, (width // 2, height // 2), 50, color, thickness)
    elif emotion == 'fear':
        shake_x = np.random.randint(-5, 5)
        shake_y = np.random.randint(-5, 5)
        cv2.rectangle(frame, (50 + shake_x, 50 + shake_y), (width - 50 + shake_x, height - 50 + shake_y), color, -1)
    elif emotion == 'disgust':
        angle = int(progress * 360)
        cv2.ellipse(frame, (width//2, height//2), (100, 50), angle, 0, 360, color, -1)
    elif emotion == 'neutral':
        y_pos = height // 2
        cv2.line(frame, (0, y_pos), (width, y_pos), color, 3)
        
    return frame

# --- 3. MULTIMODAL SYNTHETIC DATA GENERATION ---
@st.cache_resource(show_spinner=False)
def generate_multimodal_synthetic_data():
    st.info(f"Generating synthetic multimodal data using `{Config.OLLAMA_MODEL}`...")
    s_dir = Config.SYNTHETIC_DATA_DIR
    if os.path.exists(s_dir): shutil.rmtree(s_dir)
    os.makedirs(os.path.join(s_dir, 'audio'))
    os.makedirs(os.path.join(s_dir, 'video'))

    all_data = []
    utterance_id_counter = 0
    progress_bar = st.progress(0)
    status_text = st.empty()

    try: ollama.list()
    except Exception:
        st.error("Ollama server not running. Please start Ollama and refresh.")
        return None

    for i, emotion in enumerate(Config.EMOTION_MAP.keys()):
        status_text.text(f"Generating data for emotion: {emotion.capitalize()}...")
        prompt = f"Generate 10 distinct, short, realistic conversational sentences expressing '{emotion}'."
        
        try:
            response = ollama.generate(model=Config.OLLAMA_MODEL, prompt=prompt)
            sentences = [re.sub(r'^\d+\.\s*', '', s.strip()) for s in response['response'].split('\n') if s.strip()]

            for sent in sentences:
                all_data.append({'Utterance': sent, 'Emotion': emotion, 'Utterance_ID': utterance_id_counter})
                
                tts = gTTS(text=sent, lang='en')
                audio_path = os.path.join(s_dir, 'audio', f"{utterance_id_counter}.wav")
                tts.save(audio_path)

                # <<< CHANGE: Generate dynamic frames instead of static color >>>
                frames = [generate_emotion_frame(emotion, i) for i in range(16)]
                video_path = os.path.join(s_dir, 'video', f"{utterance_id_counter}.mp4")
                
                clip = ImageSequenceClip(frames, fps=8)
                audio_clip = AudioFileClip(audio_path)
                
                clip.audio = audio_clip
                
                clip.write_videofile(video_path, codec='libx264', audio_codec='aac', logger=None, threads=4)

                utterance_id_counter += 1
        except Exception as e:
            st.warning(f"Skipping emotion '{emotion}' due to error: {e}")
        progress_bar.progress((i + 1) / len(Config.EMOTION_MAP))

    df = pd.DataFrame(all_data)
    if df.empty:
        st.error("Data generation failed for all emotions. Please check your Ollama and library installations.")
        return None
        
    df.to_csv(os.path.join(s_dir, 'data.csv'), index=False)
    status_text.success("Synthetic data generation complete!")
    return s_dir

# --- 4. DATASET & DATALOADER ---
class ACAFDataset(Dataset):
    def __init__(self, data_dir, tokenizer, audio_processor, video_processor, max_len):
        self.data = pd.read_csv(os.path.join(data_dir, 'data.csv'))
        self.data_dir = data_dir
        self.tokenizer, self.audio_processor, self.video_processor = tokenizer, audio_processor, video_processor
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['Utterance']
        encoding = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        
        audio_path = os.path.join(self.data_dir, 'audio', f"{row['Utterance_ID']}.wav")
        waveform, sr = torchaudio.load(audio_path)
        
        target_sr = self.audio_processor.feature_extractor.sampling_rate
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

        waveform = waveform.squeeze()
        if waveform.shape[0] > Config.AUDIO_MAX_LENGTH:
            waveform = waveform[:Config.AUDIO_MAX_LENGTH]
        else:
            padding = Config.AUDIO_MAX_LENGTH - waveform.shape[0]
            waveform = F.pad(waveform, (0, padding))
        
        audio_input = self.audio_processor(waveform, sampling_rate=target_sr, return_tensors='pt').input_values.squeeze(0)

        video_path = os.path.join(self.data_dir, 'video', f"{row['Utterance_ID']}.mp4")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < 16:
            ret, frame = cap.read()
            if not ret: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        while len(frames) < 16: frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        processed_frames = self.video_processor(images=frames[:16], return_tensors="pt").pixel_values

        emotion_id = Config.EMOTION_MAP[row['Emotion']]
        return {
            'input_ids': encoding['input_ids'].squeeze(0), 'attention_mask': encoding['attention_mask'].squeeze(0),
            'audio_input': audio_input, 'video_input': processed_frames,
            'emotion': torch.tensor(emotion_id, dtype=torch.long)
        }

# --- 5. MODEL ARCHITECTURE ---
class ACAFModel(nn.Module):
    def __init__(self, config):
        super(ACAFModel, self).__init__()
        self.config = config
        self.roberta = XLMRobertaModel.from_pretrained(config.TEXT_MODEL)
        self.text_fc = nn.Linear(config.EMBED_DIM, config.FUSION_DIM)
        self.wav2vec = Wav2Vec2Model.from_pretrained(config.AUDIO_MODEL)
        self.audio_fc = nn.Linear(config.EMBED_DIM, config.FUSION_DIM)
        self.video_model = ViTModel.from_pretrained(config.VIDEO_MODEL)
        self.video_fc = nn.Linear(config.EMBED_DIM, config.FUSION_DIM)
        self.bilstm = nn.LSTM(input_size=config.FUSION_DIM * 3, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)
        self.fusion_fc1 = nn.Linear(128 * 2, 256)
        self.fusion_fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.emotion_classifier = nn.Linear(128, config.NUM_EMOTIONS)

    def forward(self, input_ids, attention_mask, audio_input, video_input):
        text_out = self.roberta(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        text_emb = F.relu(self.text_fc(text_out))
        
        audio_out = self.wav2vec(audio_input).last_hidden_state.mean(dim=1)
        audio_emb = F.relu(self.audio_fc(audio_out))
        
        batch_size, num_frames, C, H, W = video_input.shape
        video_input = video_input.view(batch_size * num_frames, C, H, W)
        video_out = self.video_model(pixel_values=video_input).last_hidden_state[:, 0, :]
        video_out = video_out.view(batch_size, num_frames, -1).mean(dim=1)
        video_emb = F.relu(self.video_fc(video_out))

        fused_emb = torch.cat([text_emb, audio_emb, video_emb], dim=1)
        lstm_out, _ = self.bilstm(fused_emb.unsqueeze(1))
        lstm_out = lstm_out.squeeze(1)
        x = self.dropout(F.relu(self.fusion_fc1(lstm_out)))
        x = self.dropout(F.relu(self.fusion_fc2(x)))
        emotion_logits = self.emotion_classifier(x)
        return emotion_logits

# --- 6. TRAINING & VALIDATION SCRIPT ---
def train_and_validate(config, data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.write(f"Using device: {device}")

    tokenizer = XLMRobertaTokenizer.from_pretrained(config.TEXT_MODEL)
    audio_processor = Wav2Vec2Processor.from_pretrained(config.AUDIO_MODEL)
    video_processor = AutoImageProcessor.from_pretrained(config.VIDEO_MODEL)
    
    dataset = ACAFDataset(data_dir, tokenizer, audio_processor, video_processor, config.MAX_LEN)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    model = ACAFModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    st.write("Starting training & validation...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    chart_placeholder = st.empty()

    for epoch in range(config.EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            emotion_logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['audio_input'].to(device), batch['video_input'].to(device))
            loss = criterion(emotion_logits, batch['emotion'].to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                emotion_logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['audio_input'].to(device), batch['video_input'].to(device))
                preds = torch.argmax(emotion_logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['emotion'].cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='weighted')
        status_text.text(f'Epoch {epoch+1}/{config.EPOCHS} | Validation F1-Score: {f1:.4f}')
        progress_bar.progress((epoch + 1) / config.EPOCHS)
        
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=list(Config.EMOTION_MAP.keys()), 
            labels=list(range(Config.NUM_EMOTIONS)),
            output_dict=True, 
            zero_division=0
        )
        df_report = pd.DataFrame(report).transpose().round(2)
        chart_placeholder.dataframe(df_report)

    st.success("Training complete!")
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    st.success(f"Model saved to {config.MODEL_SAVE_PATH}")
    st.balloons()

# --- 7. STREAMLIT DASHBOARD ---
def run_dashboard():
    st.set_page_config(page_title="ACAF Dashboard", layout="wide", initial_sidebar_state="expanded")
    st.title("🧠 Affective Contextual Analysis Framework (ACAF)")
    st.markdown("An end-to-end system for multimodal emotion analysis using synthetic data and live video processing.")

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @st.cache_resource
    def load_all_models():
        model = ACAFModel(config).to(device)
        if os.path.exists(config.MODEL_SAVE_PATH):
            model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        model.eval()
        tokenizer = XLMRobertaTokenizer.from_pretrained(config.TEXT_MODEL)
        audio_processor = Wav2Vec2Processor.from_pretrained(config.AUDIO_MODEL)
        video_processor = AutoImageProcessor.from_pretrained(config.VIDEO_MODEL)
        return model, tokenizer, audio_processor, video_processor

    model, tokenizer, audio_processor, video_processor = load_all_models()

    st.sidebar.header("Controls")
    app_mode = st.sidebar.radio("Choose Mode", ["Live Inference", "Train Model"])

    if app_mode == "Train Model":
        st.header("🤖 Model Training")
        st.info("This will generate a new multimodal dataset and train the model from scratch.")
        if st.button("Start Full Training Pipeline", type="primary"):
            data_dir = generate_multimodal_synthetic_data()
            if data_dir:
                train_and_validate(config, data_dir)

    elif app_mode == "Live Inference":
        st.header("🎬 Live Multimodal Inference")
        if not os.path.exists(config.MODEL_SAVE_PATH):
             st.error(f"Model not found at '{config.MODEL_SAVE_PATH}'. Please train the model first.")
             st.stop()
        
        uploaded_file = st.file_uploader("Upload a short video file (.mp4, .mov, .avi)", type=['mp4', 'mov', 'avi'])
        
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
            tfile.write(uploaded_file.read())
            
            st.video(tfile.name)

            if st.button("Analyze Video", type="primary"):
                with st.spinner("Processing video... This may take a moment."):
                    waveform, sr = torchaudio.load(tfile.name)
                    
                    target_sr = audio_processor.feature_extractor.sampling_rate
                    if sr != target_sr:
                        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
                    
                    waveform = waveform.squeeze()
                    if waveform.shape[0] > Config.AUDIO_MAX_LENGTH:
                        waveform = waveform[:Config.AUDIO_MAX_LENGTH]
                    else:
                        padding = Config.AUDIO_MAX_LENGTH - waveform.shape[0]
                        waveform = F.pad(waveform, (0, padding))
                    
                    audio_input = audio_processor(waveform, sampling_rate=target_sr, return_tensors='pt').input_values.to(device)

                    cap = cv2.VideoCapture(tfile.name)
                    frames = []
                    while len(frames) < 16:
                        ret, frame = cap.read()
                        if not ret: break
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cap.release()
                    while len(frames) < 16: frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                    video_input = video_processor(images=frames[:16], return_tensors="pt").pixel_values.to(device)

                    text_input = "Analyzing spoken content from video."
                    encoding = tokenizer(text_input, max_length=config.MAX_LEN, padding='max_length', truncation=True, return_tents='pt')
                    input_ids = encoding['input_ids'].to(device)
                    attention_mask = encoding['attention_mask'].to(device)

                    with torch.no_grad():
                        emotion_logits = model(input_ids, attention_mask, audio_input, video_input)
                    
                    probs = F.softmax(emotion_logits, dim=1).squeeze().cpu().numpy()
                    emotion_id = np.argmax(probs)
                    pred_emotion = config.ID_TO_EMOTION.get(emotion_id, "Unknown")

                    st.subheader("Analysis Results")
                    st.metric("Dominant Emotion", pred_emotion.capitalize())
                    
                    fig = px.bar(x=[p * 100 for p in probs], y=[config.ID_TO_EMOTION[i].capitalize() for i in range(len(probs))],
                                 orientation='h', labels={'x': 'Probability (%)', 'y': 'Emotion'}, title="Emotion Probabilities")
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
            os.unlink(tfile.name)

if __name__ == '__main__':
    run_dashboard()
