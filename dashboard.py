"""
Streamlit dashboard for ACAF real-time analysis.
"""
import streamlit as st
import torch
import plotly.express as px
from framework import ACAFModel, ACAFDataset
import pandas as pd

st.title("Affective Contextual Analysis Framework (ACAF) Dashboard")
st.write("Analyze emotions, intensity, sentiment drift, and intent in conversations.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ACAFModel(num_emotions=4, num_intents=5).to(device)
model.load_state_dict(torch.load('acaf_model.pth'))
model.eval()

data_dir = 'meld_data'
dataset = ACAFDataset(data_dir, split='test', modalities=['text', 'audio', 'video'])
sample = dataset[0]

with torch.no_grad():
    data = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
    emotion_logits, intensity, drift, intent_logits, _, attn_weights = model(data)
    emotion = torch.argmax(emotion_logits, dim=1).item()
    intent = torch.argmax(intent_logits, dim=1).item()
    emotion_map = {0: 'Anger', 1: 'Joy', 2: 'Sadness', 3: 'Neutral'}
    intent_map = {0: 'Question', 1: 'Statement', 2: 'Request', 3: 'Command', 4: 'Clarification'}
    st.write(f"Predicted Emotion: {emotion_map[emotion]}")
    st.write(f"Emotion Intensity: {intensity.item():.2f}")
    st.write(f"Sentiment Drift: {drift.item():.2f}")
    st.write(f"Predicted Intent: {intent_map[intent]}")

    # Interactive visualization
    fig = px.imshow(attn_weights[0].cpu().numpy(), color_continuous_scale='Viridis')
    fig.update_layout(title='Cross-Modal Attention Weights')
    st.plotly_chart(fig)

st.write("Use cases: Healthcare (patient monitoring), Education (student engagement), HR (interview analysis), Gaming (narrative adaptation).")
