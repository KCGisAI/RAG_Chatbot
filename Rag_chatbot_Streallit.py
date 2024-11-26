import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_message


st.set_page_config(page_title = "의정부 맛집 추천", page_icon="💕")
st.title("🍲의정부 맛집 챗봇🍲")
load_dotenv()

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder="오늘은 어떤 것을 먹을까요?"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role":"user", "content":user_question})

    with st.spinner("답변 생성 중"):
        ai_message = get_ai_message(user_question)
        with st.chat_message("AI"):
            st.write(ai_message)
        st.session_state.message_list.append({"role":"user", "content": "메시지"})

# streamlit run Rag_chatbot_Streallit.py