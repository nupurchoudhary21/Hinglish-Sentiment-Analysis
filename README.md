# Hinglish Sentiment Analyzer

A Gradio web app for classifying Hindi-English (Hinglish) text into **negative**, **neutral**, or **positive** sentiment using a TF-IDF vectorizer and Linear SVM.

## Run locally

1. Use Python 3.11.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   python app.py
   ```

4. Open the local URL printed in the terminal (normally `http://127.0.0.1:7860`).

## Live application demo at:
https://hinglish-sentiment-analysis-mcdw.onrender.com/

<img width="1252" height="395" alt="image" src="https://github.com/user-attachments/assets/195d21e3-973e-4425-bcea-47501d275440" />


Free web services may sleep after inactivity and take time to wake up. A public URL is stable, but free hosting is not guaranteed to be continuously awake.

## Repository contents

- `app.py`: deployment-ready Gradio app.
- `models/`: trained LinearSVC model and TF-IDF vectorizer.
- `notebooks/`: original project notebooks.

The raw dataset is intentionally not included. Do not upload it unless you have permission to redistribute it.

## Model score note

The classifier is `LinearSVC`. Its decision scores are transformed with softmax only to show relative class scores in the interface. They are **not calibrated probabilities**.

The original interface reports 74.22% test accuracy and 0.7412 test F1. These are the values recorded in the supplied notebook/interface; they have not been independently re-evaluated as part of deployment preparation.
