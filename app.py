import os
import re

import emoji
import ftfy
import gradio as gr
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

# The training notebook uses this order for the encoded target labels.
LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL_DISPLAY = {
    "negative": "😞 Negative",
    "neutral": "😐 Neutral",
    "positive": "😊 Positive",
}


def preprocess_text(text: str) -> str:
    """Apply the same text-cleaning steps used by the original interface."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = ftfy.fix_text(text)
    text = emoji.demojize(text, delimiters=(" EMOJI_", "_ "))
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = text.lower()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^a-z0-9\u0900-\u097F\s_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sentiment(text: str):
    if not text or not text.strip():
        return {"Status": "Please enter some text."}

    cleaned = preprocess_text(text)
    if not cleaned:
        return {"Status": "No usable text remained after preprocessing."}

    features = vectorizer.transform([cleaned])
    predicted_id = int(model.predict(features)[0])

    if predicted_id < 0 or predicted_id >= len(LABEL_NAMES):
        return {"Status": f"Unexpected model label: {predicted_id}"}

    # LinearSVC decision scores are not calibrated probabilities.
    # Softmax is used only to display relative class scores.
    decision = np.asarray(model.decision_function(features)).reshape(-1)
    decision = decision - np.max(decision)
    relative_scores = np.exp(decision)
    relative_scores = relative_scores / relative_scores.sum()

    scores = {
        LABEL_DISPLAY[LABEL_NAMES[int(class_id)]]: float(relative_scores[int(class_id)])
        for class_id in model.classes_
    }
    return scores


EXAMPLES = [
    ["Yaar aaj ka din bohot achha tha, sach mein amazing feel ho raha hai!"],
    ["Delivery 3 ghante late aayi aur order bhi galat tha. Worst service ever!"],
    ["Aaj office mein meeting thi, phir ghar aa gaya. Normal day tha."],
    ["Naya phone liya, camera quality is absolutely mind-blowing!"],
    ["Exam mein fail ho gaya, itni mehnat bekar gayi. So disappointed."],
    ["Maine chai banayi aur TV dekha, kuch khaas nahi tha aaj."],
    ["Bhai ka result aya first division! Itna proud feel ho raha hoon!"],
    ["Internet service bohot slow hai, kaam karna impossible ho gaya."],
    ["Iss movie ne toh dil jeet liya, absolutely loved every scene!"],
    ["price dekh ke dimag ghoom gaya, itna mehenga product hai yeh?!"],
]

CSS = """
#title { text-align: center; }
#subtitle { text-align: center; }
.gradio-container { max-width: 900px !important; margin: auto !important; }
"""


with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Hinglish Sentiment Analyzer") as demo:
    gr.Markdown("# 🇮🇳 Hinglish Sentiment Analyzer", elem_id="title")
    gr.Markdown(
        "**Hindi–English code-mixed sentiment analysis**  \n"
        "Model: Linear SVM + TF-IDF (unigrams and bigrams)",
        elem_id="subtitle",
    )
    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=3):
            input_text = gr.Textbox(
                label="Enter Hinglish text",
                placeholder="Type something in Hinglish…",
                lines=4,
            )
            with gr.Row():
                analyze = gr.Button("Analyze sentiment", variant="primary")
                clear = gr.Button("Clear")
        with gr.Column(scale=2):
            output = gr.Label(
                label="Prediction and relative class scores",
                num_top_classes=3,
            )

    gr.Markdown("### Try an example")
    gr.Examples(examples=EXAMPLES, inputs=input_text)

    gr.Markdown(
        "> **About the scores:** The model is a LinearSVC. The displayed class scores "
        "are softmax-transformed decision scores for comparison; they are **not "
        "calibrated probabilities**."
    )

    with gr.Accordion("How it works", open=False):
        gr.Markdown(
            "1. Fix text encoding and convert emojis to text.  \n"
            "2. Remove URLs and mentions, normalize repeated characters, and clean text.  \n"
            "3. Convert the text into TF-IDF features.  \n"
            "4. Predict one of three classes: negative, neutral, or positive."
        )

    with gr.Accordion("Model details", open=False):
        gr.Markdown(
            "- Classifier: Linear Support Vector Classifier (LinearSVC)\n"
            "- Features: TF-IDF, unigram and bigram features\n"
            "- Maximum TF-IDF features configured during training: 20,000\n"
            "- Dataset: Hinglish sentiment dataset used in the accompanying notebooks\n"
            "- Reported test accuracy in the original interface: 74.22%\n"
            "- Reported test F1 in the original interface: 0.7412"
        )

    analyze.click(predict_sentiment, inputs=input_text, outputs=output)
    input_text.submit(predict_sentiment, inputs=input_text, outputs=output)
    clear.click(lambda: ("", None), outputs=[input_text, output])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
