import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from pojinega import classify_sentiment
import pandas as pd
from datetime import datetime
import os
from transformers import pipeline

# ▼ APIキーを入力
GOOGLE_API_KEY = ""

# LLM初期化
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)

#日記の保存    
def save_diary_log(user_input, label, score):
        LOG_FILE = "diary_log.csv"
        if not os.path.exists(LOG_FILE):
            pd.DataFrame(columns=["timestamp", "user_input", "label", "score"]).to_csv(LOG_FILE, index=False, encoding='utf-8-sig')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = pd.DataFrame([[timestamp, user_input, label, score]], columns=["timestamp", "user_input", "label", "score"])
        new_data.to_csv(LOG_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

# ページ設定
st.set_page_config(page_title="日記チャットボット", layout="centered")
st.title("📘 今日の気分を話そう")

# セッションで履歴管理
if "messages" not in st.session_state:
    st.session_state.messages = []
if "diary_input" not in st.session_state:
    st.session_state.diary_input = ""

# 入力欄（チャット入力）
user_input = st.chat_input("今日あったことを教えてね")

if user_input:
    #フラグ
    st.session_state.save_next_input = True
    # 入力保存（チャット履歴用と日記処理用）
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.diary_input = user_input  # ← ボタン用に一時保存

    # チャット表示（ユーザー）
    with st.chat_message("user"):
        st.markdown(user_input)

    # Gemini応答（通常チャット）

    with st.spinner("考え中..."):
        response = llm.invoke(st.session_state.messages)

    st.session_state.messages.append(response)
    with st.chat_message("assistant"):
        st.markdown(response.content)

# ネガポジ判定（BERT）
    with st.expander("🧠 この入力の感情分析（BERTによる）"):
        label, score = classify_sentiment(user_input)
        label_ja = "ポジティブ" if label == "positive" else "ネガティブ"
        st.info(f"📊 感情分類：**{label_ja}（スコア: {score:.2f}**")
    
    if st.session_state.get("save_next_input"):  # フラグがあれば保存
        save_diary_log(user_input, label, score)



# ▼ 「日記を書く」ボタンが押されたとき：user_input を独立処理
if st.button('日記を書く'):
    if st.session_state.diary_input:
        with st.spinner("日記を要約中..."):
            diary_result = llm.invoke([
                HumanMessage(content=f"以下は今日の日記です。気分を一言でまとめてください：\n\n{st.session_state.diary_input}")
            ])
        st.success("✅ 日記の感情まとめ：")
        st.write(diary_result.content)
    else:
        st.warning("日記を書くには、まずチャット欄に文章を入力してください。")

# ▼ 「タスクを登録する」ボタンは空処理のまま
if st.button('タスクを登録する'):
    st.write('タスク登録機能は今後追加予定です。')
