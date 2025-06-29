import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from pojinega import classify_sentiment

# ▼ APIキーを入力
GOOGLE_API_KEY = "AIzaSyAFEhzZzt2pSa7z3mH75RTcUje22v6Sf3U"

# LLM初期化
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)

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
        sentiment = classify_sentiment(user_input)
        st.info(f"📊 感情分類：**{sentiment}**")



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
