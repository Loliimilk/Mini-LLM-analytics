import streamlit as st
import pandas as pd
import os
import glob
import time
from agent import get_analysis_agent, run_secure_query

# Абсолютный путь к папке charts/ рядом с app.py — не зависит от CWD при запуске
CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")

st.set_page_config(page_title="Data Analyzer", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    .stChatFloatingInputContainer { bottom: 20px; }

    /* --- Сайдбар: опускаем текст --- */
    section[data-testid="stSidebar"] .stFileUploader {
        margin-top: 40px;
    }

    /* --- Заголовок предпросмотра --- */
    .preview-title {
        margin-top: 2.5rem;
        margin-bottom: 0.5rem;
        font-size: 1rem;
        font-weight: 500;
    }

    /* --- "Загрузите файл..." на одном уровне с предпросмотром --- */
    .upload-hint {
        margin-top: 2.5rem;
        padding: 1rem 1.25rem;
        background-color: rgba(28, 131, 225, 0.08);
        border: 1px solid rgba(28, 131, 225, 0.35);
        border-radius: 0.5rem;
        color: inherit;
        font-size: 1rem;
    }

    /* --- Чат: аватар и контент по центру по вертикали --- */
    div[data-testid="stChatMessage"] {
        align-items: center !important;
    }

    /* Текст внутри bubble строго по центру */
    div[data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"] [data-testid="stVerticalBlock"],
    div[data-testid="stChatMessage"] [data-testid="stMarkdown"],
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin: 0 !important;
    }

    div[data-testid="stChatMessage"] p {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.5;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .chat-text {
        margin: 0 !important;
        line-height: 1.5;
        overflow-wrap: anywhere;
    }

    /* --- Кнопка ↻: круглая, размером с аватар (~2rem) --- */
    section[data-testid="stMain"] .stButton button {
        width: 2rem !important;
        height: 2rem !important;
        min-width: 2rem !important;
        min-height: 2rem !important;
        padding: 0 !important;
        border-radius: 50% !important;
        line-height: 1 !important;
        font-size: 1rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* Колонка с кнопкой ↻ — центрировать по вертикали со строкой сообщения */
    section[data-testid="stMain"] div[data-testid="column"]:has(.stButton) {
        display: flex;
        align-items: center;
    }

    /* Графики в чате компактнее */
    div[data-testid="stChatMessage"] [data-testid="stImage"] img,
    div[data-testid="stChatMessage"] img {
        max-width: 600px !important;
        height: auto !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Папка для графиков ---
os.makedirs(CHARTS_DIR, exist_ok=True)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = None
if "last_failed" not in st.session_state:
    st.session_state.last_failed = False


def load_data(uploaded_file):
    try:
        name = uploaded_file.name.lower()
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        df = pd.read_csv(uploaded_file, sep=";")
        if df.shape[1] == 1:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=",")
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки «{uploaded_file.name}»: {e}")
        return None


def run_query(dfs, df_names, prompt, chat_text_only, user_context=""):
    for tmp in glob.glob("temp_chart_*.png"):
        os.remove(tmp)

    agent = get_analysis_agent(dfs, df_names, user_context=user_context)
    result = run_secure_query(agent, prompt, chat_text_only)
    answer = result["output"]

    img_paths = []
    timestamp = int(time.time())
    for tmp in sorted(glob.glob("temp_chart_*.png")):
        idx = len(img_paths) + 1
        final = os.path.join(CHARTS_DIR, f"chart_{timestamp}_{idx}.png")
        os.rename(tmp, final)
        img_paths.append(final)

    return answer, img_paths


def do_query(dfs, df_names, prompt, user_context=""):
    history = []
    for m in st.session_state.messages[:-1]:
        n = len(m.get("images", []))
        marker = f" [было создано графиков: {n}]" if n else ""
        history.append({"role": m["role"], "content": (m["content"] or "") + marker})

    try:
        answer, img_paths = run_query(dfs, df_names, prompt, history, user_context)
        new_msg = {"role": "assistant", "content": answer}
        if img_paths:
            new_msg["images"] = img_paths
        st.session_state.messages.append(new_msg)
        st.session_state.last_failed = False
    except Exception as e:
        st.session_state.last_failed = True
        st.session_state.last_error = str(e)


# --- SIDEBAR ---
with st.sidebar:
    st.header("Данные")
    uploaded_files = st.file_uploader(
        "Загрузите файлы",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="CSV (разделитель ; или ,) и Excel. Можно загрузить несколько файлов.",
    )

    user_context = st.text_area(
        "Контекст / на что обратить внимание",
        placeholder="Например: свяжи таблицы по столбцу 'имя', "
                    "сравни продажи с зарплатами сотрудников.",
        height=120,
        help="Опционально. Будет передано агенту как приоритеты анализа.",
    )

    st.markdown("---")
    if st.button("Очистить чат", use_container_width=True):
        for msg in st.session_state.messages:
            for img in msg.get("images", []):
                if os.path.exists(img):
                    os.remove(img)
        st.session_state.messages = []
        st.session_state.last_prompt = None
        st.session_state.last_failed = False
        st.rerun()

# --- ОСНОВНОЙ ЭКРАН ---
if uploaded_files:
    # Загружаем все файлы
    dfs_named = []
    for f in uploaded_files:
        df = load_data(f)
        if df is not None:
            dfs_named.append((f.name, df))

    if dfs_named:
        dfs = [df for _, df in dfs_named]
        df_names = [name for name, _ in dfs_named]

        # --- ПРЕВЬЮ ---
        st.markdown('<div class="preview-title">Предпросмотр данных (6 строк):</div>', unsafe_allow_html=True)
        if len(dfs_named) == 1:
            st.dataframe(dfs[0].head(6), use_container_width=True)
        else:
            tabs = st.tabs([name for name, _ in dfs_named])
            for tab, (name, df) in zip(tabs, dfs_named):
                with tab:
                    st.dataframe(df.head(6), use_container_width=True)
        st.markdown("---")

        # --- ЧАТ ---
        for i, msg in enumerate(st.session_state.messages):
            is_last = (i == len(st.session_state.messages) - 1)
            is_failed_user_msg = (
                is_last
                and msg["role"] == "user"
                and st.session_state.last_failed
            )

            if is_failed_user_msg:
                bubble_col, btn_col = st.columns([0.94, 0.06], vertical_alignment="center")
                with bubble_col:
                    with st.chat_message("user"):
                        st.markdown(f'<div class="chat-text">{msg["content"]}</div>', unsafe_allow_html=True)
                with btn_col:
                    if st.button("↻", key="retry_btn"):
                        with st.spinner(""):
                            do_query(dfs, df_names, st.session_state.last_prompt, user_context)
                        st.rerun()

                err = st.session_state.get("last_error", "")
                if err:
                    st.error(f"Ошибка: {err}")

            else:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "user":
                        # Пользовательский текст: plain-text обёртка для стилей
                        st.markdown(f'<div class="chat-text">{msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        # Ответ агента: стандартный markdown (таблицы, буллеты, заголовки)
                        st.markdown(msg["content"])

                    for img in msg.get("images", []):
                        if os.path.exists(img):
                            st.image(img, use_container_width=True)

        # --- INPUT ---
        if prompt := st.chat_input("Спросите о данных..."):
            st.session_state.last_prompt = prompt
            st.session_state.last_failed = False
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(f'<div class="chat-text">{prompt}</div>', unsafe_allow_html=True)

            with st.spinner(""):
                do_query(dfs, df_names, prompt, user_context)
            st.rerun()

else:
    st.markdown(
        '<div class="upload-hint">Загрузите файл в боковой панели.</div>',
        unsafe_allow_html=True,
    )
