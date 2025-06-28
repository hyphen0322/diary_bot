import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer

model = AutoModelForSequenceClassification.from_pretrained('koheiduck/bert-japanese-finetuned-sentiment') 
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
nlp = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)

input_df = pd.read_csv('名言.csv')
results = []
for row_i in input_df.itertuples():
    #row_i[1]：タイトル、row_i[2]：キャラ、row_i[3]：セリフ
    analyze_result = nlp(row_i[3])
    judge = analyze_result[0]['label']
    score = analyze_result[0]['score']
    results.append([row_i[1], row_i[2], row_i[3], judge, score])

colmn_name = ['タイトル','キャラ','セリフ','ポジネガ判定結果','確信度']
result_df = pd.DataFrame(results, columns=colmn_name)
result_df.to_csv("ポジネガ判定結果.csv", index=False, encoding="utf-8")
