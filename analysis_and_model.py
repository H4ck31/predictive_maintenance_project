# analysis_and_model.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from ucimlrepo import fetch_ucirepo
import io

# --- Настройка страницы (уже задана в app.py, но можно дополнять) ---
st.title("📊 Анализ данных и обучение модели")
st.markdown("Это приложение позволяет загрузить датасет, обучить модель машинного обучения для предсказания отказа оборудования и сделать прогноз на новых данных.")

# --- 1. Загрузка данных ---
st.header("1. Загрузка данных")

# Создаем сессионное состояние для хранения данных и модели
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Вариант 1: Загрузка через интерфейс
uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("Данные успешно загружены!")
        st.write("Первые 5 строк загруженных данных:")
        st.dataframe(data.head())
        st.session_state.data = data
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")

# Вариант 2: Загрузка из репозитория UCI, если файл не загружен
if st.session_state.data is None:
    if st.button("Загрузить демонстрационный датасет из UCI Repository"):
        with st.spinner("Загрузка данных..."):
            try:
                dataset = fetch_ucirepo(id=601)
                # Объединяем признаки и целевую переменную
                data_uci = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
                st.session_state.data = data_uci
                st.success("Демонстрационный датасет успешно загружен!")
                st.write("Первые 5 строк данных:")
                st.dataframe(data_uci.head())
            except Exception as e:
                st.error(f"Ошибка при загрузке данных из UCI: {e}")

# Если данные есть, показываем информацию о них и предлагаем предобработку
if st.session_state.data is not None:
    data = st.session_state.data
    st.header("2. Предобработка данных")

    # Информация о данных
    with st.expander("Информация о данных"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Размер данных:**")
            st.write(data.shape)
        with col2:
            st.write("**Типы данных:**")
            st.write(data.dtypes.value_counts())
        st.write("**Статистика:**")
        st.dataframe(data.describe())

    # Кнопка для выполнения предобработки
    if st.button("Выполнить предобработку"):
        with st.spinner("Выполняется предобработка..."):
            # 1. Удаление ненужных столбцов
            # В датасете есть UDI, Product ID, а также столбцы с типами отказов. Нам нужен только Machine failure.
            columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
            # Проверяем, какие столбцы есть в данных
            cols_to_drop = [col for col in columns_to_drop if col in data.columns]
            if cols_to_drop:
                data = data.drop(columns=cols_to_drop)
                st.info(f"Удалены столбцы: {', '.join(cols_to_drop)}")

            # 2. Преобразование категориальной переменной 'Type'
            if 'Type' in data.columns:
                le = LabelEncoder()
                data['Type'] = le.fit_transform(data['Type'])
                st.info("Категориальная переменная 'Type' преобразована в числовой формат.")

            # 3. Проверка на пропуски
            if data.isnull().sum().sum() == 0:
                st.success("Пропущенные значения не обнаружены.")
            else:
                st.warning("Обнаружены пропуски! Будет выполнено удаление строк с пропусками.")
                data = data.dropna()
                st.info(f"После удаления пропусков осталось {data.shape[0]} записей.")

            # 4. Разделение на признаки (X) и целевую переменную (y)
            target_col = 'Machine failure' if 'Machine failure' in data.columns else 'Target'
            if target_col not in data.columns:
                st.error(f"Столбец '{target_col}' не найден. Проверьте названия.")
                st.stop()
            X = data.drop(columns=[target_col])
            y = data[target_col]

            # 5. Масштабирование признаков
            numerical_features = ['Air temperature [K]', 'Process temperature [K]',
                                  'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
            # Проверка, какие числовые признаки есть в данных
            num_cols_to_scale = [col for col in numerical_features if col in X.columns]
            if num_cols_to_scale:
                scaler = StandardScaler()
                X[num_cols_to_scale] = scaler.fit_transform(X[num_cols_to_scale])
                st.success(f"Признаки {', '.join(num_cols_to_scale)} масштабированы.")
                st.session_state.scaler = scaler

            # Сохраняем обработанные данные в сессию
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.data_processed = data

            st.success("Предобработка завершена!")
            st.write("**Обработанные данные (первые 5 строк):**")
            st.dataframe(X.head())
            st.write(f"**Распределение целевой переменной:**\n{y.value_counts()}")

    # --- 3. Обучение модели ---
    if 'X' in st.session_state:
        X = st.session_state.X
        y = st.session_state.y
        st.header("3. Обучение модели")

        # Разделение на обучающую и тестовую выборки
        test_size = st.slider("Размер тестовой выборки", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random State", value=42, step=1)

        # Выбор модели
        model_name = st.selectbox(
            "Выберите модель:",
            ("Logistic Regression", "Random Forest", "XGBoost")
        )

        # Параметры для Random Forest
        if model_name == "Random Forest":
            n_estimators = st.slider("Количество деревьев (n_estimators)", 10, 200, 100, 10)
        elif model_name == "XGBoost":
            n_estimators = st.slider("Количество итераций (n_estimators)", 10, 200, 100, 10)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)

        if st.button("Обучить модель"):
            with st.spinner("Обучение модели..."):
                # Разделение данных
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                st.info(f"Размер обучающей выборки: {X_train.shape[0]}, тестовой: {X_test.shape[0]}")

                # Выбор и обучение модели
                if model_name == "Logistic Regression":
                    model = LogisticRegression(random_state=random_state, max_iter=1000)
                elif model_name == "Random Forest":
                    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                else: # XGBoost
                    import xgboost as xgb
                    model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                              random_state=random_state, use_label_encoder=False,
                                              eval_metric='logloss')
                model.fit(X_train, y_train)
                st.session_state.model = model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test

                st.success("Модель обучена!")

        # --- 4. Оценка модели ---
        if st.session_state.model is not None:
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            st.header("4. Оценка модели")

            # Предсказания
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
            with col2:
                st.metric("ROC-AUC", f"{roc_auc:.4f}")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            # ROC Curve
            st.subheader("ROC-кривая")
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC-кривая')
            ax.legend()
            st.pyplot(fig)

            # --- 5. Предсказание на новых данных ---
            st.header("5. Предсказание на новых данных")
            st.markdown("Введите значения признаков для предсказания:")

            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                with col1:
                    type_input = st.selectbox("Тип продукта (Type)", options=["L", "M", "H"])
                    air_temp = st.number_input("Air temperature [K]", value=300.0)
                    process_temp = st.number_input("Process temperature [K]", value=310.0)
                with col2:
                    rotational_speed = st.number_input("Rotational speed [rpm]", value=1500.0)
                    torque = st.number_input("Torque [Nm]", value=40.0)
                    tool_wear = st.number_input("Tool wear [min]", value=100.0)

                submitted = st.form_submit_button("Предсказать")

                if submitted:
                    # Создаем DataFrame из введенных данных
                    input_df = pd.DataFrame({
                        'Type': [type_input],
                        'Air temperature [K]': [air_temp],
                        'Process temperature [K]': [process_temp],
                        'Rotational speed [rpm]': [rotational_speed],
                        'Torque [Nm]': [torque],
                        'Tool wear [min]': [tool_wear]
                    })

                    # Применяем те же преобразования, что и к обучающим данным
                    # 1. Преобразуем Type
                    le = LabelEncoder()
                    # Для корректности нужно обучить LabelEncoder на всех возможных значениях
                    le.fit(['L', 'M', 'H'])
                    input_df['Type'] = le.transform(input_df['Type'])

                    # 2. Масштабируем
                    if st.session_state.scaler is not None:
                        numerical_cols = ['Air temperature [K]', 'Process temperature [K]',
                                          'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
                        input_df[numerical_cols] = st.session_state.scaler.transform(input_df[numerical_cols])

                    # 3. Предсказываем
                    prediction = st.session_state.model.predict(input_df)
                    prediction_proba = st.session_state.model.predict_proba(input_df)[0, 1]

                    if prediction[0] == 1:
                        st.error(f"⚠️ **Предсказание: ОТКАЗ** с вероятностью {prediction_proba:.2f}")
                    else:
                        st.success(f"✅ **Предсказание: РАБОТАЕТ** с вероятностью отказа {prediction_proba:.2f}")