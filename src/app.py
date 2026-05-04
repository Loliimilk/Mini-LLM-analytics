import streamlit as st
import pandas as pd
from agent import get_analysis_agent, run_secure_query

st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("ИИ-Аналитик (Агент)")

uploaded_file = st.sidebar.file_uploader("Загрузите данные", type=["csv", "xlsx", "xls"])

@st.cache_data 
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file, sep=None, engine='python', encoding='utf-8')
        else:
            return pd.read_excel(file)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, sep=None, engine='python', encoding='cp1251')
    except Exception as e:
        st.error(f"Не удалось прочитать файл: {e}")
        return None

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.subheader("Предпросмотр данных")
        st.dataframe(df.head(5))
        
        user_query = st.text_input("Задайте вопрос по данным:", 
                                   placeholder="Например: Выведи названия колонок и количество строк")
        
        if user_query:
            if st.button("Запустить анализ"):
                with st.spinner("Агент думает и пишет код..."):
                    try:
                        agent = get_analysis_agent(df)
                        result = run_secure_query(agent, user_query)
                        
                        st.success("Готово!")
                        st.markdown("### Ответ:")
                        st.write(result["output"])
                    except Exception as e:
                        st.error(f"Ошибка при работе агента: {e}")
else:
    st.info("Загрузите CSV или Excel файл на боковой панели.")