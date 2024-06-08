import streamlit as st
from langserve.client import RemoteRunnable
import json
from streamlit_navigation_bar import st_navbar
import update_base 
from pyvis.network import Network
import networkx as nx


def format_patient_info(patient_info):
    patient_info_str = f"""
                    **Рост:** {patient_info['height']} см
                    **Вес:** {patient_info['weight']} кг
                    **Возраст:** {patient_info['age']}
                    **ИМТ:** {patient_info['bmi']}
                    **Глюкоза в крови:** {patient_info['blood_glucose']} ммоль/л
                    **Гликированный гемоглобин:** {patient_info['hba1c']}%
                    **ЧСС:** {patient_info['heart_rate']} удары/мин
                    **АД:** {patient_info['blood_pressure']}
                    **Жалобы:** {patient_info['complaints']}
                    **Описание:** {patient_info['description']}
                    **Основной Диагноз:** {patient_info['main_diagnosis']}
                    **Дополнительно:** {patient_info['additional_info']}
                    **Анамнез:** {patient_info['history']}
                    """
    return patient_info_str

def patient_info_app():
    # Настройка дизайна страницы

    # Инициализация или обновление session_state
    if 'submit_clicked' not in st.session_state:
        st.session_state['submit_clicked'] = False

    if 'documents' not in st.session_state:
        st.session_state['documents'] = []

    if 'triplets' not in st.session_state:
        st.session_state['triplets'] = None

    if 'patient_info' not in st.session_state:
        st.session_state['patient_info'] = {}

    if 'uploaded' not in st.session_state:
        st.session_state['uploaded'] = None

    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = ""

    if 'model_params' not in st.session_state:
        st.session_state['model_params'] = {}
            
    model_name = {
        'gpt-3.5-turbo': 'OpenAI GPT-3.5',
        'gpt-4-turbo': 'OpenAI GPT-4',
        'gpt-4o': 'OpenAI GPT-4o',
        'claude-3-opus-20240229': 'Claude Opus',
        'llama-3-sonar-large-32k-chat':'Perplexity Sonar Large'
    }

    status_error = {
        1:'Для данного пациента не был найден подходящий документ'
    }

    def format_func(option):
        return model_name[option]

    selected_model = st.sidebar.selectbox("Выберите модель:", options=model_name.keys(), format_func=format_func)

    st.session_state['api_key'] = st.sidebar.text_input(f"API ключ для {model_name[selected_model]}", value=st.session_state['api_key'], type='password')

    st.session_state['model_params']['temperature'] = st.sidebar.slider(label='Температура',min_value=0.1,max_value=1.0,value=0.5,step=0.05)
    st.session_state['model_params']['top_p'] = st.sidebar.slider(label='Top-p',min_value=0.1,max_value=1.0,value=0.9,step=0.05)

    # Загрузка JSON файла
    uploaded_file = st.file_uploader("Загрузите файл с данными пациента", type=['json'])
    if uploaded_file is not None and st.session_state['uploaded'] != uploaded_file:
        data = json.load(uploaded_file)
        st.session_state['patient_info'] = data
        st.session_state['uploaded'] = uploaded_file
        st.experimental_rerun()

    if st.session_state['uploaded'] is not None:
        st.success("Файл загружен")

    # Форма для ввода информации о пациенте
    with st.form(key='patient_info_form'):
        st.header("Форма для информации о пациенте")

        # Использование сохраненных значений или установка None
        patient_info = st.session_state.get('patient_info', {})
        height = st.number_input("Рост (см)", value=patient_info.get('height', None), min_value=0, max_value=250, step=1, format="%d")
        weight = st.number_input("Вес (кг)", value=patient_info.get('weight', None), min_value=0, max_value=300, step=1, format="%d")
        age = st.number_input("Возраст", value=patient_info.get('age', None), min_value=0, max_value=120, step=1, format="%d")
        bmi = st.number_input("ИМТ (индекс массы тела)", value=patient_info.get('bmi', None), min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
        blood_glucose = st.number_input("Глюкоза в крови (ммоль/л)", value=patient_info.get('blood_glucose', None), min_value=0.0, max_value=50.0, step=0.1, format="%.1f")
        hba1c = st.number_input("Гликированный гемоглобин (%)", value=patient_info.get('hba1c', None), min_value=0.0, max_value=20.0, step=0.1, format="%.1f")
        heart_rate = st.number_input("ЧСС (удары в минуту)", value=patient_info.get('heart_rate', None), min_value=0, max_value=300, step=1, format="%d")
        blood_pressure = st.text_input("АД (артериальное давление, мм рт.ст.)", value=patient_info.get('blood_pressure', ''))
        complaints = st.text_input("Жалобы", value=patient_info.get('complaints', ''))
        description = st.text_area("Описание", value=patient_info.get('description', ''))
        main_diagnosis = st.text_input("Основной Диагноз", value=patient_info.get('main_diagnosis', ''))
        additional_info = st.text_area("Дополнительно", value=patient_info.get('additional_info', ''))
        history = st.text_area("Анамнез", value=patient_info.get('history', ''))

        # Кнопка отправки формы
        submit_button = st.form_submit_button(label='Отправить')

    # Логика обработки нажатия кнопки "Отправить"
    if submit_button:
        error = False
        if not (height and weight and age and bmi and blood_glucose and hba1c and heart_rate and blood_pressure and complaints and description and main_diagnosis and additional_info and history):
            st.error("Пожалуйста, заполните все поля.")
            error = True
        elif st.session_state['api_key'] == '':
            st.error(f'Пожалуйста, введите ваш api ключ для {model_name[selected_model]}')
            error = True
        else:
            patient_info = {
                "height": height,
                "weight": weight,
                "age": age,
                "bmi": bmi,
                "blood_glucose": blood_glucose,
                "hba1c": hba1c,
                "heart_rate": heart_rate,
                "blood_pressure": blood_pressure,
                "complaints": complaints,
                "description": description,
                "main_diagnosis": main_diagnosis,
                "additional_info": additional_info,
                "history": history
            }
        if not error:
            st.session_state['patient_info'] = patient_info  # Сохранение информации о пациенте

            with st.spinner('Пожалуйста, подождите... Идет обработка данных'):
                
                patient_info_dict = {
                    'height':height,
                    'weight':weight,
                    'age':age,
                    'bmi':bmi,
                    'blood_glucose':blood_glucose,
                    'hba1c':hba1c,
                    'heart_rate':heart_rate,
                    'blood_pressure':blood_pressure,
                    'complaints':complaints,
                    'description':'description',
                    'main_diagnosis':main_diagnosis,
                    'additional_info':additional_info,
                    'history':history
                }

                def generate_response():
                    client = RemoteRunnable('http://localhost:8000/med/')

                    key = st.session_state['api_key']
                    model_params = st.session_state['model_params']
                    output = client.invoke({'patient_info': patient_info_dict, 'model': selected_model, 'model_params':model_params, 'key': key})
                    status = output.get('status', 'Не удалось получить ответ от сервера')
                    if(status != 0):
                        raise Exception(status_error[status])
                    recoms = output.get('recoms', 'Не удалось получить ответ от сервера')
                    triplets = output.get('triplets', 'Не удалось получить ответ от сервера')
                    return recoms, triplets

                try:
                    generated_text, triplets = generate_response()
                except Exception as error:
                    st.error(error)
                    return

                st.session_state['generated_text'] = generated_text
                st.session_state['triplets'] = triplets  # Сохранение документов в session_state

            # Отображение результатов
            st.success("Информация успешно обработана")
            st.write(format_patient_info(patient_info))
            st.markdown("### Рекомендации")
            st.markdown(generated_text)
            st.session_state['submit_clicked'] = True

    elif st.session_state['submit_clicked']:
        # Отображение сохраненной информации и результатов, если форма была отправлена ранее
        st.success("Информация успешно обработана")
        st.write(format_patient_info(st.session_state['patient_info']))
        st.markdown("### Рекомендации")
        st.markdown(st.session_state['generated_text'])

    # Отображение документов
    if st.session_state['triplets']:
        if st.button('Показать сущности'):
            st.info(st.session_state['triplets'])

def download_doc_app():
    st.title("Knowledge Base Updater")
    
    if st.button("Обновить базу знаний"):
        st.success(update_base.get_todo())
    
    # Загружаем список документов (пример, нужно заменить на реальную загрузку документов)
    documents = update_base.get_docs()
    
    selected_document = st.selectbox("Выберите документ для отображения графа", documents)
    selected_document_index = documents.index(selected_document)
    if selected_document:
        with open(f'data/KR_{update_base.documents[selected_document_index]}_rec.txt','r',encoding='utf-8') as f:
            text = f.read()
        entities = update_base.convert_entity_data(text)
        update_base.draw_graph(entities)
        
        HtmlFile = open("graph.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        st.components.v1.html(source_code, height=500)


def main():

    pages = {
        'Генерация рекомендаций':patient_info_app,
        'Документы':download_doc_app
    }

    page = st_navbar(['Генерация рекомендаций','Документы'])
    pages[page]()

if __name__ == '__main__':
    main()
