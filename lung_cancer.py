import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Загружаем модели
model1 = joblib.load('MR_lung.pkl')
model2 = joblib.load('NB_lung.pkl')
model3 = joblib.load('SVM_lung.pkl')

# Маппинг английских столбцов на русский
column_translation = {
    'Age': 'Возраст',
    'Gender': 'Пол',
    'Air Pollution': 'воздействия загрязнения воздуха',
    'Alcohol use': 'употребления алкоголя',
    'Dust Allergy': 'аллергии на пыль',
    'OccuPational Hazards': 'производственной опасности',
    'Genetic Risk': 'генетического риска',
    'chronic Lung Disease': 'хронических заболеваний легких',
    'Balanced Diet': 'сбалансированности питания',
    'Obesity': 'ожирения',
    'Smoking': 'курения',
    'Passive Smoker': 'пассивного курения',
    'Chest Pain': 'боли в груди',
    'Coughing of Blood': 'кашля с кровью',
    'Fatigue': 'усталости',
    'Weight Loss': 'потери веса',
    'Shortness of Breath': 'одышки',
    'Wheezing': 'свистящего дыхания',
    'Swallowing Difficulty': 'трудностей при глотании',
    'Clubbing of Finger Nails': 'утолщения кончиков пальцев',
    'Frequent Cold': 'частых простуд',
    'Dry Cough': 'сухого кашля',
    'Snoring': 'храпа'
}

# Загружаем минимальные и максимальные значения для каждого столбца
min_max_values = {
    'Age': (14, 73),
    'Gender': (1, 2),
    'Air Pollution': (1, 8),
    'Alcohol use': (1, 8),
    'Dust Allergy': (1, 8),
    'OccuPational Hazards': (1, 8),
    'Genetic Risk': (1, 7),
    'chronic Lung Disease': (1, 7),
    'Balanced Diet': (1, 7),
    'Obesity': (1, 7),
    'Smoking': (1, 8),
    'Passive Smoker': (1, 8),
    'Chest Pain': (1, 9),
    'Coughing of Blood': (1, 9),
    'Fatigue': (1, 9),
    'Weight Loss': (1, 8),
    'Shortness of Breath': (1, 9),
    'Wheezing': (1, 8),
    'Swallowing Difficulty': (1, 8),
    'Clubbing of Finger Nails': (1, 9),
    'Frequent Cold': (1, 7),
    'Dry Cough': (1, 7),
    'Snoring': (1, 7)
}

# Заголовок приложения
st.title('Прогноз развития рака легких')

# Слайдер для возраста
age = st.slider(
    'Укажите Ваш возраст',
    min_value=min_max_values['Age'][0],
    max_value=min_max_values['Age'][1],
    value=min_max_values['Age'][0],
    step=1
)

input_data = {'Age': age}  # Сохраняем значение возраста

# Пол
gender = st.radio('Укажите Ваш пол', options=['М', 'Ж'])
input_data['Gender'] = 1 if gender == 'М' else 2

# Описание признаков
st.subheader('Оцените уровень следующих признаков')

# Для остальных признаков показываем слайдеры
for column in min_max_values.keys():
    if column != 'Age' and column != 'Gender':
        translated_column = column_translation[column]  # Получаем переведенное название
        min_val, max_val = min_max_values[column]  # Получаем минимальные и максимальные значения
        
        input_data[column] = st.slider(
            f'Уровень {translated_column}:',
            min_value=min_val,
            max_value=max_val,
            value=min_val,
            step=1
        )

# Создаем DataFrame для входных данных
input_df = pd.DataFrame([input_data])

# Маппинг для классов
label_mapping = {0: 'Низкий', 1: 'Средний', 2: 'Высокий'}

if st.button('Предсказать развитие рака легких'):
    # Прогнозирование и получение вероятностей для каждой модели
    prediction1 = model1.predict(input_df)
    probability1 = model1.predict_proba(input_df)

    prediction2 = model2.predict(input_df)
    probability2 = model2.predict_proba(input_df)

    prediction3 = model3.predict(input_df)
    probability3 = model3.predict_proba(input_df)

    # Вывод прогнозов для каждой модели
    st.subheader('Результаты прогноза')
    for i, (pred, prob) in enumerate(zip(
            [prediction1, prediction2, prediction3],
            [probability1, probability2, probability3])):
        model_name = f'Модель {i + 1}'
        st.write(f"### {model_name}")
        st.write(f'Прогноз: {label_mapping[pred[0]]}')
        st.write(f'Вероятности для классов (Низкий, Средний, Высокий): '
                 f'{prob[0][0] * 100:.2f}%, {prob[0][1] * 100:.2f}%, {prob[0][2] * 100:.2f}%')

    # Объединение вероятностей всех моделей
    combined_probabilities = np.mean([probability1, probability2, probability3], axis=0)

    # Проверка на размерность
    if combined_probabilities.shape[1] == 3:
        prob_0 = np.round(combined_probabilities[0][0] * 100, 2)
        prob_1 = np.round(combined_probabilities[0][1] * 100, 2)
        prob_2 = np.round(combined_probabilities[0][2] * 100, 2)
    else:
        # Если одна из моделей вернула только один класс, обработаем это
        prob_0 = prob_1 = prob_2 = 0  # или какое-то другое значение по умолчанию

    # Определение итогового прогноза на основе максимальной вероятности
    final_label_idx = np.argmax(combined_probabilities)
    final_label = label_mapping[final_label_idx]

    st.markdown(f"<h2 style='font-size: 30px;'>Общий прогноз: {final_label}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='font-size: 25px;'>Общая вероятность для классов (Низкий, Средний, Высокий): "
                f"{prob_0}%, {prob_1}%, {prob_2}%</h3>", unsafe_allow_html=True)