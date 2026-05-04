import streamlit as st
import pandas as pd
import os
import time
from agent import get_analysis_agent, run_secure_query

st.set_page_config(page_title="LLM Data Analyzer", layout="wide")
st.title("Интеллектуальный анализ данных")

if "charts_history" not in st.session_state:
    st.session_state.charts_history = []

def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, sep=";")
        return df
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
        return None

with st.sidebar:
    st.header("Настройки")
    uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")
    
    if st.button("Очистить историю графиков"):
        for img in st.session_state.charts_history:
            if os.path.exists(img):
                os.remove(img)
        st.session_state.charts_history = []
        st.rerun()

col_main, col_history = st.columns([3, 1])

if uploaded_file:
    df = load_data(uploaded_file)
    
    with col_main:
        if df is not None:
            st.subheader("Предпросмотр данных")
            st.dataframe(df.head(5))
            
            user_query = st.text_input("Задайте вопрос или попросите график:", 
                                       placeholder="Например: Построй гистограмму цен")
            
            if user_query and st.button("Запустить анализ"):
                if os.path.exists("temp_chart.png"):
                    os.remove("temp_chart.png")
                    
                with st.spinner("Агент анализирует..."):
                    try:
                        agent = get_analysis_agent(df)
                        result = run_secure_query(agent, user_query)
                        
                        st.markdown("### Ответ:")
                        st.write(result["output"])
                        
                        if os.path.exists("temp_chart.png"):
                            # Сохраняем его в историю с уникальным именем
                            timestamp = int(time.time())
                            new_filename = f"chart_{timestamp}.png"
                            os.rename("temp_chart.png", new_filename)
                            st.session_state.charts_history.append(new_filename)
                            
                            st.image(new_filename, caption=f"Визуализация для: {user_query}")
                            
                    except Exception as e:
                        st.error(f"Ошибка: {e}")

    with col_history:
        st.subheader("История графиков")
        if not st.session_state.charts_history:
            st.info("Здесь появятся ваши графики")
        else:
            for i, img_path in enumerate(reversed(st.session_state.charts_history)):
                with st.expander(f"График #{len(st.session_state.charts_history) - i}"):
                    st.image(img_path)
                    with open(img_path, "rb") as file:
                        st.download_button(
                            label="Скачать",
                            data=file,
                            file_name=img_path,
                            mime="image/png",
                            key=f"dl_{img_path}"
                        )
else:
    with col_main:
        st.info("Пожалуйста, загрузите файл в меню слева.")