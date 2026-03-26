# pneumonia-detector
# 🫁 PneumoScan AI

An AI-powered pneumonia detection web app built with FastAI and Streamlit.

## What it does
Upload a chest X-Ray image and the model predicts whether it shows
signs of pneumonia — with confidence scores and detailed analytics.

## Tech Stack
- Python, FastAI, ResNet-34
- Streamlit
- Plotly
- Kaggle Chest X-Ray Dataset

## How to run locally
1. Clone this repo
2. Install dependencies: `pip install fastai streamlit plotly pillow`
3. Download the dataset from Kaggle: chest-xray-pneumonia
4. Run training: `python train_model.py`
5. Launch app: `streamlit run app.py`

## Results
- Validation accuracy: 81%
- Training images: 5,216
- Model: ResNet-34 fine-tuned with transfer learning
