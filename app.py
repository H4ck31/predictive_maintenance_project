# app.py
import streamlit as st

# --- Настройка страницы ---
st.set_page_config(
    page_title="Predictive Maintenance App",
    page_icon="⚙️",
    layout="wide"
)

# --- Навигация ---
page_analis = st.Page("analysis_and_model.py", title="📊 Анализ и модель")
page_presentation = st.Page("presentation.py", title="📽️ Презентация")

current_page = st.navigation([page_analis, page_presentation], position="sidebar")

# --- Запуск текущей страницы ---
current_page.run()
