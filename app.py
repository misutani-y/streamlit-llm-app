import os
from pathlib import Path
from functools import lru_cache

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
EXPERT_ROLES = {
    "保険・ライフプランニングの専門家": """
あなたは日本の保険・ライフプランニング領域の専門家です。
ユーザーの状況を引き出しつつ、
1) 前提整理 2) 選択肢提示（長短所）3) 実行アクション 4) 注意点（法令/コンプライアンス）
の順で、わかりやすく日本語で助言してください。必要なら公的保険制度も平易に説明します。
""",
    "AI開発の専門家": """
あなたはAI/LLMアーキテクトです。LangChainやRAG、評価・運用まで精通しています。
ユーザーの課題に対し、
1) 目的の明確化 2) 推奨アーキテクチャ 3) 手順/コード例 4) リスク・代替案
を、箇条書き中心で日本語簡潔に回答してください。必要に応じてベストプラクティスも示します。
"""
}
DEFAULT_ROLE_KEY = "保険・ライフプランニングの専門家"


def run_expert_llm(user_text: str, selected_role_key: str) -> str:
    system_message = EXPERT_ROLES.get(selected_role_key, EXPERT_ROLES[DEFAULT_ROLE_KEY])

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input_text}"),
    ])

    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"input_text": user_text})

@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")

# === Streamlit UI ===
st.set_page_config(page_title="専門家に聞いてみよう", page_icon="🤖", layout="centered")
st.title("🤖 専門家に聞いてみよう")

with st.expander("このアプリについて（概要・操作方法）", expanded=True):
    st.markdown(
        """
        **概要**
        - **ボタン**で専門家を選ぶと、**システムメッセージが自動で切り替わり**、回答の観点が変わります。

        **操作手順**
        1. 専門家を選択
        2. 下の入力欄に質問や文章を入力
        3. 「送信」を押すと、数秒後に回答が表示されます
        """
    )

# ロール選択（ラジオボタン）
role_key = st.radio("専門家を選択", options=list(EXPERT_ROLES.keys()), index=0, horizontal=False)

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

# print(result.content) # This line was causing an error as 'result' was not defined
