# presentation.py
import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("📽️ Презентация проекта")

    # Содержание презентации в формате Markdown
    # Каждый слайд отделяется "---"
    presentation_markdown = """
# Прогнозирование отказов оборудования
## Бинарная классификация для предиктивного обслуживания

---

## Введение

- **Задача:** Предсказать отказ оборудования (Target = 1) или его отсутствие (Target = 0).
- **Актуальность:** Своевременное предсказание отказов позволяет снизить время простоя оборудования и затраты на обслуживание.
- **Датасет:** AI4I 2020 Predictive Maintenance Dataset (UCI Repository, id=601).

---

## Датасет

- **Количество записей:** 10 000.
- **Основные признаки:**
    - Тип продукта (Type: L, M, H)
    - Air temperature, Process temperature (K)
    - Rotational speed (rpm)
    - Torque (Nm)
    - Tool wear (min)
- **Целевая переменная:** `Machine failure` (1 - отказ, 0 - работа)

---

## Этапы работы

1. **Загрузка и предобработка данных**
    - Удаление лишних столбцов (UDI, Product ID, типы отказов).
    - Преобразование категориальных признаков (Type).
    - Масштабирование числовых признаков.
2. **Разделение данных на обучающую и тестовую выборки** (80/20).
3. **Обучение моделей:**
    - Logistic Regression
    - Random Forest
    - XGBoost
4. **Оценка моделей** (Accuracy, Confusion Matrix, ROC-AUC).
5. **Создание веб-приложения** на Streamlit.

---

## Результаты обучения

| Модель | Accuracy | ROC-AUC |
|--------|----------|---------|
| Logistic Regression | ~0.97 | ~0.98 |
| Random Forest | ~0.99 | ~0.99 |
| XGBoost | ~0.99 | ~0.99 |

*Наилучшая модель: Random Forest / XGBoost*

---

## Streamlit-приложение

- **Страница "Анализ и модель":**
    - Загрузка данных (CSV или из UCI).
    - Предобработка одним кликом.
    - Обучение модели с выбором параметров.
    - Визуализация метрик и графиков.
    - Интерфейс для предсказания на новых данных.
- **Страница "Презентация":**
    - Интерактивная презентация проекта (вы ее сейчас видите).

---

## Заключение и улучшения

- **Что сделано:**
    - Разработано приложение для предиктивного обслуживания.
    - Проведен анализ данных и обучение моделей.
    - Получены высокие метрики качества (AUC > 0.98).
- **Возможные улучшения:**
    - Использовать более сложные модели (например, нейронные сети).
    - Добавить настройку гиперпараметров.
    - Реализовать мониторинг модели в реальном времени.

---

## Спасибо за внимание!

## Вопросы?

    """

    # Настройка презентации
    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"], index=1)
        height = st.number_input("Высота слайдов (px)", value=500, step=50)
        transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"], index=0)
        plugins = st.multiselect("Плагины", ["highlight", "notes", "search", "zoom"], default=["highlight"])

    # Отображение презентации
    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "^--$"},
    )

# Вызываем функцию страницы
presentation_page()