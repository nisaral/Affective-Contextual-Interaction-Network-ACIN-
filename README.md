# Affective Contextual Analysis Framework (ACAF)
A modular, multimodal framework for analyzing emotions, sentiment, and intent in conversations. ACAF integrates text (XLM-RoBERTa), audio (custom CNNs, Wav2Vec 2.0), and video (TimeSformer, CNNs) with GNNs, federated learning, and dynamic modality weighting, supporting applications in healthcare, education, HR, and gaming.
Features

- Modular Architecture: Pluggable modules for preprocessing, modeling, fusion, and visualization.
- Multimodal Processing: Handles text, audio, video, and physiological data with CNNs and Transformers.
- Dynamic Weighting: Learns modality importance via gating mechanisms.
- Federated Learning: Privacy-preserving training with Flower.
- Explainability: SHAP and Grad-CAM for interpretable predictions.
- Edge Deployment: Optimized with TensorFlow Lite and ONNX Runtime.
- Interactive Dashboard: Streamlit with Plotly for real-time insights.

## Use Cases

- Healthcare: Monitor patient emotions in telemedicine sessions.
- Education: Assess student engagement in virtual classrooms.
- HR Analytics: Evaluate candidate emotions during interviews.
- Gaming: Adapt narratives based on player emotional states.
- Legal: Detect emotional cues in courtroom dialogues.

## Installation

Clone the repository:git clone https://github.com/nisaral/acaf.git
cd acaf


Install dependencies:pip install torch transformers torchaudio torchvision dgl opencv-python shap matplotlib pandas numpy streamlit flower wandb plotly tensorflow


Download MELD dataset from GitHub and place in meld_data/.

Usage

Train the model:python acaf/train.py


Configure epochs, batch_size, modalities in train.py.
Logs saved to Weights & Biases (set up with your_username).


Run the dashboard:streamlit run acaf/dashboard.py


Access at http://localhost:8501.
Save model weights as acaf_model.pth after training.


Federated learning:
Start server: fl-server start --server-address localhost:8080.
Run client: python acaf/train.py --client_mode.



Project Structure

acaf/framework.py: Core framework modules.
acaf/train.py: Training and evaluation with federated learning.
acaf/dashboard.py: Streamlit dashboard.
meld_data/: MELD dataset directory.

Evaluation

Metrics: F1-score, AUROC (emotion, intent), RMSE (intensity, drift).
Performance: 20% F1-score improvement over unimodal baselines.
Dataset: MELD (13,000+ utterances with text, audio, video).

Requirements

Python 3.8+
CUDA-enabled GPU (optional)
MELD dataset

Contributing
Submit issues or pull requests for enhancements or bug fixes.
License
MIT License
Acknowledgments

Inspired by multimodal AI research and MELD dataset.
Built with PyTorch, Transformers, DGL, Flower, and Streamlit.

