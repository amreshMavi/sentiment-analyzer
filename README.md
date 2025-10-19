# Sentiment-Analyzer

## Project 1: Sentiment Analysis from Social Media Screenshots

## 📌 Project Overview -----
This project extracts text from uploaded screenshots of social media posts and performs sentiment analysis. It eliminates the need for direct API access, making it useful for analyzing public sentiment visually captured from platforms like Twitter, Reddit, and Instagram.

## 🧰 Features

Upload image files directly in Jupyter Notebook.

Extract text using OCR.

Clean and preprocess text.

Perform sentiment analysis (positive/negative/neutral).

Display results with confidence scores and visualizations.


## 📁 Links for dataset:- Text Data - https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format


## 🛠️ Dependencies

Make sure you have the following installed:
python>=3.8
jupyter
pytesseract
Pillow
opencv-python
regex
textblob
matplotlib
ipywidgets
numpy

Cmd to install all at once:

pip install jupyter pytesseract Pillow opencv-python regex textblob matplotlib ipywidgets numpy


## 📌 Also install the OCR engine:

Download Tesseract OCR - https://github.com/tesseract-ocr/tesseract

Add its path to your system environment variables if needed.


## 📂 Project Structure

sentiment_screenshot_project/
│── notebooks/
│   └── 01_sentiment_analyzer.ipynb
│── dataset/
│   └── csv file
│── requirements.txt
│── README.md


## ▶️ How to Run
1 - Install all dependencies.

2 - Launch Jupyter Notebook: cmd - jupyter notebook

3 - Open 01_sentiment_analyzer.ipynb

4 - Run the cells to upload an image and get sentiment results.


## 🧪 Example Output

Extracted Text

Sentiment Classification: Positive / Negative / Neutral

Confidence Score

Sentiment Visualization (Bar Chart)


## 🚀 Future Enhancements

Add support for multilingual OCR.

Deploy as a simple web app with Gradio or Flask.

Incorporate sarcasm or emotion detection.


## Project 2: Facial Emotion Detection


## 📌 Project Overview

This project uses a deep learning model trained on the AffectNet dataset to detect facial emotions (happy, sad, angry, surprised, fearful, neutral) from images uploaded through an interactive UI.


## 🧰 Features

Parse image + text label dataset (AffectNet format).

Train MobileNetV2-based model with TensorFlow/Keras.

Upload single or multiple images interactively.

Predict and display emotion with probability distribution.

Visualization of model confidence scores. 


## 📂 Link for dataset:- Facial Expression data - https://www.kaggle.com/datasets/kazanova/sentiment140


## 🛠️ Dependencies

python>=3.8
tensorflow>=2.12
opencv-python
Pillow
numpy
matplotlib
ipywidgets
scikit-learn

Cmd to install all at once:
pip install tensorflow opencv-python Pillow numpy matplotlib ipywidgets scikit-learn


## 📂 Project Structure

facial_emotion_detection/
│── data/
│   ├── test/
|       └── images
|       └── labels
│   ├── train/
|       └── images
|       └── labels
│   ├── valid/
|       └── images
|       └── labels
│── notebooks/
│   └── 02_human_expressions.ipynb
│── requirements.txt
│── README.md


## ▶️ How to Run

1 - Install all dependencies.

2 - Place training/testing data in the data/ folder.

3 - Launch Jupyter Notebook: jupyter notebook

4 - Open 02_human_expressions.ipynb.

5 - After ensuring gpu is available for use, Train the model on AffectNet (long runtime)

6 - Upload an image using the widget to get predictions.


## 🧪 Example Output

Top predicted emotion (e.g., “Happy”)

Probability distribution for each class

Probability bar chart visualization


## 🚀 Future Enhancements

Expand emotion categories (e.g., contempt, disgust).

Deploy as real-time webcam app.

Combine text + facial emotion for richer sentiment analysis.
