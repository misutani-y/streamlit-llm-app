import os
from functools import lru_cache

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
EXPERT_ROLES = {
# === 指定の条件を満たす関数 ===
# ・「入力テキスト」と「ラジオボタンでの選択値」を引数として受け取り
# ・LLM からの回答を戻り値として返す


def run_expert_llm(user_text: str, selected_role_key: str) -> str:
system_message = EXPERT_ROLES.get(selected_role_key, EXPERT_ROLES[DEFAULT_ROLE_KEY])


prompt = ChatPromptTemplate.from_messages([
("system", system_message),
("human", "{input_text}"),
])


chain = prompt | get_llm() | StrOutputParser()
return chain.invoke({"input_text": user_text})


# === Streamlit UI ===
st.set_page_config(page_title="LangChain Expert App", page_icon="🤖", layout="centered")
st.title("🤖 LangChain Expert App")


with st.expander("このアプリについて（概要・操作方法）", expanded=True):
st.markdown(
"""
**概要**
- 入力フォームのテキストを LangChain 経由で LLM に投げ、**回答を画面に表示**します。
- **ラジオボタン**で専門家ロール（A/B）を選ぶと、**システムメッセージが自動で切り替わり**、回答の観点が変わります。


**操作手順**
1. 専門家ロール（A または B）を選択
2. 下の入力欄に質問や文章を入力
3. 「送信」を押すと、数秒後に回答が表示されます
"""
)


# ロール選択（ラジオボタン）
role_key = st.radio("専門家ロールを選択", options=list(EXPERT_ROLES.keys()), index=0, horizontal=False)


# 入力フォーム
user_text = st.text_area("入力テキスト", placeholder="ここに質問や文章を入力してください…", height=180)


col1, col2 = st.columns([1, 1])
with col1:
submitted = st.button("送信", type="primary")
with col2:
show_prompt = st.toggle("デバッグ: システムメッセージを表示", value=False)


# API キー確認
if not os.getenv("OPENAI_API_KEY"):
st.warning("OPENAI_API_KEY が環境変数に設定されていません。実行前に設定してください。", icon="⚠️")


# 実行
if submitted:
if not user_text.strip():
st.error("入力テキストを入力してください。")
else:
try:
if show_prompt:
with st.expander("実際に使われるシステムメッセージ", expanded=False):
st.code(EXPERT_ROLES[role_key], language="markdown")


with st.spinner("LLM に問い合わせ中…"):
answer = run_expert_llm(user_text, role_key)


st.subheader("回答")
st.markdown(answer)
except Exception as e:
st.exception(e)

print(result.content)