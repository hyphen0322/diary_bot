# pojinega.py
from transformers import pipeline

# モデルの読み込み（初期化は1度だけに）
classifier = pipeline("sentiment-analysis",
                      model="christian-phu/bert-finetuned-japanese-sentiment")

def classify_sentiment(text: str) -> str:
    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]

    return label, score