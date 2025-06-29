import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
import pandas as pd
from datetime import datetime
import os
from transformers import pipeline

# ▼ ここにあなたのAPIキーを入力してください（安全な運用は後述）
GOOGLE_API_KEY = "AIzaSyAyDSsqG_N6G7RlqqK34S3d02mH5uhimAM"

# Geminiモデルを使うチャットボットの初期化
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)

#感情分析
def my_sentiment_analyzer(text):
    classifier = pipeline("sentiment-analysis",
    model="christian-phu/bert-finetuned-japanese-sentiment")
    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]

    if label == "positive":
        print("ポジティブなテキストです！")
    elif label == "negative":
        print("ネガティブなテキストです！")
    else:
        print("ニュートラルなテキストです！")
        
    return label, score

#日記の保存    
def save_diary_log(user_input, label, score):
        LOG_FILE = "diary_log.csv"
        if not os.path.exists(LOG_FILE):
            pd.DataFrame(columns=["timestamp", "user_input", "label", "score"]).to_csv(LOG_FILE, index=False, encoding='utf-8-sig')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = pd.DataFrame([[timestamp, user_input, label, score]], columns=["timestamp", "user_input", "label", "score"])
        new_data.to_csv(LOG_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
 
# Streamlit アプリのタイトル
st.set_page_config(page_title="日記チャットボット", layout="centered")
st.title("📘 今日の気分を話そう")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "save_next_input" not in st.session_state:
    st.session_state.save_next_input = False   

if st.button('日記を書く'):
    st.session_state.save_next_input = True
    

if st.button('タスクを登録する'):
    st.write('ボタン2がクリックされました！')
# セッション内でメッセージ履歴を保持
if "messages" not in st.session_state:
    st.session_state.messages = []

# これまでのメッセージを表示
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# 入力欄（Streamlitのチャット用UI）
if user_input := st.chat_input("今日あったことを教えてね"):
    # ユーザーの発言を記録・表示
    human_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(human_msg)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Geminiで応答を生成
    with st.spinner("考え中..."):
        response = llm.invoke(st.session_state.messages)

    # AIの返答を記録・表示
    st.session_state.messages.append(response)
    with st.chat_message("assistant"):
        st.markdown(response.content)

    if st.session_state.get("save_next_input"):  # フラグがあれば保存
        label, score = my_sentiment_analyzer(user_input)
        save_diary_log(user_input, label, score)
        st.success("日記が保存されました！")
        st.session_state.save_next_input = False