import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# ▼ ここにあなたのAPIキーを入力してください（安全な運用は後述）
GOOGLE_API_KEY = "AIzaSyAFEhzZzt2pSa7z3mH75RTcUje22v6Sf3U"

# Geminiモデルを使うチャットボットの初期化
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
 
# Streamlit アプリのタイトル
st.set_page_config(page_title="日記チャットボット", layout="centered")
st.title("📘 今日の気分を話そう")
if st.button('日記を書く'):
    st.write('ボタン1がクリックされました！')

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
