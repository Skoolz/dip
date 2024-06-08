import requests
import bs4
import regex as re
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import os
from langchain.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
import json
from operator import itemgetter
import re

from langchain_anthropic.chat_models import ChatAnthropic
from langchain_community.llms.llamacpp import LlamaCpp

from transformers import AutoTokenizer


from langchain_core.messages import SystemMessage,ChatMessage,HumanMessage,AIMessage
from huggingface_hub import notebook_login

from pyvis.network import Network
import networkx as nx

import os

documents = [
    '25',
    '261',
    '306',
    '313',
    '352',
    '359',
    '360',
    '381',
    '603',
    '654',
    '655',
    '656',
    '662',
    '664',
    '677',
    '683',
    '714',
    '724',
    '749',
    '783'
]

headers = [
            "2. Диагностика", 
            "2.1 Жалобы и анамнез", 
            "2.2 Физикальное обследование", 
            "2.4 Инструментальные диагностические исследования", 
            "2.5 Иные диагностические исследования", 
            "3. Лечение",
            "3.1",
            "3.2",
            "3.3",
            "4. Реабилитация", 
            "5. Профилактика",
            "6. Дополнительная информация, влияющая на течение и исход заболевания",
            "7. Дополнительная информация",
            "7.1",
            "7.2",
            "7.3",
            "Приложение Г"
            ]


model = None

def set_model(key):
    global model
    os.environ['OPENAI_API_KEY'] = key
    model = ChatOpenAI(model_name='gpt-4o')



#------------------------------------------------------------------------------

def clean_text(text):
    # Регулярное выражение для поиска паттернов с учетом неразрывного пробела
    pattern = r'\(J\d+(\.\d+)?\)|J\d+(\.\d+)?'
    # Замена найденных паттернов на пустую строку
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

#------------------------------------------------------------------------------

def parse_abbreviations(text):
    lines = text.splitlines()
    abbrev_dict = {}
    for line in lines:
        if ('-' in line) or ('–' in line) or ('—' in line):
            parts = re.split(r'\s*[\p{Pd}]\s', line, maxsplit=1)
            if len(parts) == 2:
                key, value = parts[0], parts[1]
                abbrev_dict[key.strip()] = value.strip()
        else:
            parts = re.split(r'\s+', line, maxsplit=1)
            if len(parts) == 2:
                key, value = parts[0], parts[1]
                abbrev_dict[key.strip()] = value.strip()
    return abbrev_dict

#------------------------------------------------------------------------------

def replace_abbreviations(text, abbreviations):
    # Создаем регулярное выражение для поиска сокращений
    # Эскейпим специальные символы в сокращениях, так как они могут влиять на регулярные выражения
    abbrev_pattern = re.compile(r'\b(' + '|'.join(re.escape(abbr) for abbr in abbreviations.keys()) + r')\b')

    # Функция для замены сокращений их полными формами
    def expand(match):
        return abbreviations[match.group(0)]

    # Замена сокращений в тексте
    return abbrev_pattern.sub(expand, text)

#------------------------------------------------------------------------------

def find_diseases_in_content(data):
    content = data[find_by_element_by_id(data,'doc_1')]['content']
    soup = bs4.BeautifulSoup(content)
    related_diseases = 'Особенности кодирования заболевания'.lower()

    current_element = None
    for h in soup.findAll('h2'):
        if(related_diseases in h.text.lower()):
            current_element = h
            break
    
    current_element = current_element.findNext()
    text = ''
    while(current_element.name != 'h2'):
        text += f'{current_element.text}\n'
        current_element = current_element.findNext()
    return text

#------------------------------------------------------------------------------

def find_by_element_by_id(data,id):
    for i in range(len(data)):
        if(data[i]['id']==id):
            return i
    return None

#------------------------------------------------------------------------------

def clean_all_text(text):
    patterns = [
        r'\[(?!TABLE|ROW|CELL|ENDTABLE)[^\]]*\]',  # ссылки на литературу
        r'\(УУР\s*[-–—]\s*[^;]+;\s*УДД\s*[-–—]\s*\d+\)',  # специфический формат с аббревиатурами
        r'УУР\s*[-–—]\s*[^;]+;\s*УДД\s*[-–—]\s*\d+\)',
        r'\s*Комментари[ий]\s*:\s*',
        #----------------------------------------------------------------
        r'Уровень\s+убедительности\s+рекомендаци[ий]\s+[-–—][\r\n]_[\w+\b+]_\s+\(Уровень\s+доказательности\s+[\s+-–—]\s+_[\w+\b+]_\)',
        r'Уровень\s+убедительности\s+рекомендаци[ий]\s+[-–—][\r\n][\w+\b+]__\s+\(Уровень\s+доказательности\s+[\s+-–—]\s+_[\w+\b+]_\)',  
        r'Уровень\s+убедительности\s+рекомендаци[ий]\s+[-–—][\r\n][\w+\b+]\s+\(Уровень\s+доказательности\s+[\s+-–—]\s+[\w+\b+]\)', 
        r'Уровень\s+убедительности\s+рекомендаци[ий]\s*[\s+-–—]?[\r\n][\w+\b+]\s*Уровень\s+доказательности\s+[\s+-–—]?\s+?[\w+\b+]\)',
        r'Уровень\s+убедительности\s+рекомендаци[ий]\s+[\w+\b+]\s+\(Уровень\s+достоверности\s+доказательств\s+[\s+-–—]?\s+?[\w+\b+]\)', 
        r'Уровень\s+убедительности\s+рекомендаци[ий]\s*[-–—]\s*[\w+\b+]\s*\(Уровень\s+доказательства\s*[-–—]\s*[\w+\b+]\)',
        r'Уровень\s+убедительности\s+рекомендаци[ий]\s+[\w+\b+]\s+\(Уровень\s+достоверности\s+доказательств\s+[\w+\b+]+\)',
        r'Уровень\s+убедительности\s+рекомендаци[ий]\s+[\w+\b+]\s+\(Уровень\s+достоверности\s+доказательств[\s*\p{Pd}][\s+-–—]?\s+[\w+\b+]\)',
        r'У\s*р\s*о\s*в\s*е\s*н\s*ь\s+у\s*б\s*е\s*д\s*и\s*т\s*е\s*л\s*ь\s*н\s*о\s*с\s*т\s*и\s+р\s*е\s*к\s*о\s*м\s*е\s*н\s*д\s*а\s*ц\s*и\s*[ий]\s+[\w+\b+]\s+\(у\s*р\s*о\s*в\s*е\s*н\s*ь\s+д\s*о\s*с\s*т\s*о\s*\s*в\s*е\s*р\s*н\s*о\s*с\s*т\s*и\s+д\s*о\s*к\s*а\s*з\s*а\s*т\s*е\s*л\s*ь\s*с\s*т\s*в\s+[\s+-–—]?\s+?[\w+\b+]\)', 
        r'Уровень\s+убедительности\s+рекомендаци[ий]\s+[\w+\b+][;\s+\p{Pd},]\s*уровень\s+достоверности\s+доказательств\s*[\s+-–—]?[\w+\b+]\s*',
        r'У\s*р\s*о\s*в\s*е\s*н\s*ь\s+у\s*б\s*е\s*д\s*и\s*т\s*е\s*л\s*ь\s*н\s*о\s*с\s*т\s*и\s+р\s*е\s*к\s*о\s*м\s*е\s*н\s*д\s*а\s*ц\s*и\s*[ий][\p{Pd}][\w+\b+][;\s+\p{Pd},]\(\s*у\s*р\s*о\s*в\s*е\s*н\s*ь\s+д\s*о\s*с\s*т\s*о\s*\s*в\s*е\s*р\s*н\s*о\s*с\s*т\s*и\s+д\s*о\s*к\s*а\s*з\s*а\s*т\s*е\s*л\s*ь\s*с\s*т\s*в\s*[\s+\p{Pd}]s*[\w+\b+]\)',  # длинное описание уровней
        r'\(Уровень\s+убедительности\s+рекомендаци[ий]\s+[\w+\b+][\p{Pd}]\s[;\s+\p{Pd},]\s*уровень\s+достоверности\s+доказательств\s*[\s+-–—]\s*[\w+\b+]\s*\)', 
        r'Уровень убедительности рекомендаци[ий][\s+\p{Pd}][\w+\b+]\s+\(\s*уровень достоверности доказательств\s*[\s+-–—]?[\w+\b+]\s*\)',
        #----------------------------------------------------------------
        r'Уровень\s+убедительности\s+рекомендаци[ий]\s+[\w+\b+]\s*\(\s*уровень\s+достоверности\s+доказательств\s*[\s+\p{Pd}]\s*[\w+\b+]', 
        r'\(Уровень\s+убедительности\s+рекомендаци[ий]\s+[\w+\b+]\s*[;\s+\p{Pd}]\s*уровень\s+достоверности\s+доказательств\s*[\s+\p{Pd}]\s*[\w+\b+]\s*', 
        r'Уровень убедительности рекомендаци[ий]\s*[\s+\p{Pd}]\s*[\w+\b+]\s*\(\s*уровень достоверности доказательств\s*[\s+\p{Pd}]\s*[\w+\b+]\s*',
        #----------------------------------------------------------------
        r'С\s*и\s*л\s*а\s+рекомендаци[ий]\s+\d+[;\s+\p{Pd},]\(\s*у\s*р\s*о\s*в\s*е\s*н\s*ь\s+д\s*о\s*с\s*т\s*о\s*\s*в\s*е\s*р\s*н\s*о\s*с\s*т\s*и\s+доказательств\s*[\s+\p{Pd}]\s+\w+\)',  # длинное описание уровней
        r'\(Сила\s+рекомендаци[ий]\s+\d+[;\s+\p{Pd},]\s*уровень\s+достоверности\s+доказательств\s*[\s+\p{Pd}]\s*\w+\s*\)', 
        r'Сила\s+рекомендаци[ий]\s*[\s+-–—]\s*\d+[;\s+\p{Pd},]\(\s*уровень достоверности доказательств\s*[\s+\p{Pd}]\s+\w+\s*\)',
        #----------------------------------------------------------------
        r'Сила\s+рекомендаци[ий]\s+\d+\s*\(\s*уровень\s+достоверности\s+доказательств\s*[\s+\p{Pd}]\s*\w+', 
        r'\(Сила\s+рекомендаци[ий]\s+\d+\s*[;\s+\p{Pd}]уровень\s+достоверности\s+доказательств\s*[\s+\p{Pd}]\s*\w+\s*', 
        r'Сила рекомендаци[ий]\s*[\s+\p{Pd}]\s*\d+\s*\(\s*уровень достоверности доказательств\s*[\s+\p{Pd}]\s*\w+\s*'
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)

    return text

#------------------------------------------------------------------------------

def parse_document(id):
    url = f'https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrec2&id={id}&ssid=undefined'
    request = requests.get(url)

    data = request.json()['obj']['sections']
    
    start = 'Диагностика'.lower()
    end = 'Дополнительная информация'.lower()
    title = 'Титульный лист'.lower()
    abbr = 'Список сокращений'.lower()
    annex_G = 'Приложение Г.'.lower()    

    start_index = None
    end_index = None
    title_index = None
    abbr_index = None
    annex_G_index = None

    for i in range(len(data)):
        d = data[i]
        name = d['title'].lower()
        if(start in name):
            start_index = i
        if(end in name):
            end_index = i
        if(title in name):
            title_index = i
        if (abbr in name):
            abbr_index = i
        if (annex_G in name):
            annex_G_index = i
        if all(x is not None for x in [end_index, start_index, title_index, abbr_index, annex_G_index]):
            break
        text = ''
        table_data = []
        for i in range(start_index,end_index+1):
            d = data[i]
            title = d['title']

            soup = bs4.BeautifulSoup(d['content'])
            
            # Получение таблиц
            table_tags = soup.find_all('table')
            t_data = []
            tables = ''
            for table in table_tags:
                rows = []
                for tr in table.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    rows.append(cells)
                line_table = "[TABLE] " + " \n".join(" | ".join(row) for row in rows)
                t_data.append(line_table) 
                table_data.append(line_table) 
            for table in t_data:
                tables += table + '\n\n' 
            #with open(f'data/KR_{id}_Tablels.txt','w', encoding='utf-8') as file:
            #    for table in table_data:
            #        file.write(table + '\n\n') 

            #сбор заголовков, текста и таблиц
            content = ' \n'.join(soup.stripped_strings)
            text += f'''
            {title}\n
            {content}\n
            {tables}\n 
            '''
            # удалить {tables}, если не нужна запись в файл с текстом
        # Достаем таблицы из проложения Г, т к там они бывают полезны 
    if annex_G_index is not None:
        d = data[annex_G_index]
        soup = bs4.BeautifulSoup(d['content'])
        table_data = []
        table_tags = soup.find_all('table')
        tables = ''
        for table in table_tags:
            rows = []
            for tr in table.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                rows.append(cells)
            line_table = "[TABLE] " + " \n ".join(" | ".join(row) for row in rows) + " [ENDTABLE]"
            table_data.append(line_table) 
        #with open(f'data/KR_{id}_annex_G_Tablels.txt','w', encoding='utf-8') as file:
        #    for table in table_data:
        #        file.write(table + '\n\n')
        #        tables += table + '\n\n'
        for table in table_data:
            tables += table + '\n\n' 
        if tables != '':
            text += 'Приложение Г\n\n' + tables 
    
    # составление словаря сокращений    
    if abbr_index is not None:
        d = data[abbr_index]
        soup = bs4.BeautifulSoup(d['content'])
        #abbr_text = soup.get_text() -- для проверок других файлов
        abbr_text = ''
        abbr_text_soup = soup.find_all('p')
        for a in abbr_text_soup:
            abbr_text = abbr_text + a.get_text() + '\n'
        # Тестировочный вывод текста 
        #with open(f'data/abbreviation/abbr_text{id}.txt', 'w', encoding='utf-8') as file:
        #   file.write(abbr_text)
        abbr_dict = parse_abbreviations(abbr_text)  
        # Тестовый вывод словаря
        #with open(f'data/abbreviation/abbr_list{id}.txt', 'w', encoding='utf-8') as file:
        #    json.dump(abbr_dict, file, ensure_ascii=False, indent=4)  # Сохранение в файл

    related_diseases_text = ''

    related_diseases_index = find_by_element_by_id(data,'doc_crat_info_1_4')

    if(related_diseases_index!=None):
        d = data[related_diseases_index]
        soup = bs4.BeautifulSoup(d['content'])
        related_diseases_text = soup.getText()
    else:
        print('Related text was not found. Using content')
        related_diseases_text = find_diseases_in_content(data)


    d = data[title_index]
    main_disease = d['data'][0]['content']
    age_content = None
    for item in d['data']:
        if item['title'] == 'Возрастная категория':
            age_content = item['content']
        break

    related_diseases_text = clean_text(related_diseases_text)

    cr_info = f'''
<id>{id}</id>:
<title>{main_disease}</title>
<age>{age_content}</age_content>
    '''
    text = clean_all_text(text)
    text = replace_abbreviations(text, abbr_dict)  

    return text,cr_info

#------------------------------------------------------------------------------

def split_document(document):
    header_pattern = re.compile(r'|'.join([fr'\b{header}\b' for header in headers]), re.IGNORECASE)
    
    # Находим позиции всех заголовков
    matches = list(header_pattern.finditer(document))
    
    # Добавляем фиктивный матч в конец документа
    fake_match = type('FakeMatch', (object,), {'start': lambda self: len(document), 'group': lambda self: 'END'})
    matches.append(fake_match())

    # Разделяем документ на части
    sections = {}
    for i in range(len(matches) - 1):
        start = matches[i].start()
        end = matches[i + 1].start()
        header = matches[i].group()
        sections[header] = document[start:end].strip()
    
    return sections

#------------------------------------------------------------------------------
# Загрузка и предварительная очистка 
def update_doc():
    documents_info = {}
    for d in documents:
        text,info = parse_document(d)
        with open(f'data/KR_{d}.txt','w+',encoding='utf-8') as f:
            f.write(text)
        documents_info[d] = info
        #print(f'{d} done')
    with open('data/documents_info.json','w+',encoding='utf-8') as f:
        json.dump(documents_info,f, ensure_ascii=False, indent=4)
    #print('json info dump done')

#------------------------------------------------------------------------------

def get_disease_title_by_id(disease_id, info):
    if disease_id in info:
        # Ищем тег <title> и возвращаем его содержимое
        match = re.search(r'<title>(.*?)</title>', info.get(disease_id))
        if match:
            return match.group(1)
    return None

#------------------------------------------------------------------------------

def get_age_by_id(disease_id, info):
    if disease_id in info:
        # Ищем тег <age> и возвращаем его содержимое
        match = re.search(r'<age>(.*?)</age>', info.get(disease_id))
        if match:
            return match.group(1)
    return None

#------------------------------------------------------------------------------

# Функции для удаления пробелов и выделения сущностей
#------------------------------------------------------------------------------

def clean_query(model, text):
    system_prompt = '''
    Ты - медицинский чатбот, который работает с медицинскими данными. 
    Тебе передается текст - глава документа клинические рекомендации. 
    Твоя задача удалить из этого текста лишние пробелы и переносы, как в словах, так и между ними. 

	Разрешено: 
	Удалять пробелы и переносы

	Запрещено: менять текст, кроме разрешенного; менять таблицы; писать слова "Выход:", "Вывод:", "Заболевание:", "Текст:"; добавлять слова

    
	Верни ровно тот же текст, но с нормальными пробелами.

    На пустой запрос не нужно никак реагировать и что-то писать, вверни такой же пустой ответ
    
    '''

    human_prompt = '''
    Текст:{text}

    Вывод:
    '''

    template = ChatPromptTemplate.from_messages([
        ('system',system_prompt),
        ('human',human_prompt)
    ])

    chain = template | model | StrOutputParser()

    output = chain.invoke(input={'text':text})

    return output

#------------------------------------------------------------------------------

def entity_query(model, text, title, age):
    system_prompt = '''
    Ты - медицинский чатбот. Твоя задача - найти в данном тексте рекомендации в виде сущностей и отношений между ними.
    Советы:
    1) Обычно условия, страдает ли пациент какими то дополнительными заболевания, курит или пьет, взрослый или нет и так далее.
    2) На вход подается текст и название заболевания. Его, как сущность разспознавать не надо. Ты должен найти зависимости между дополнительными факторами и пациентом или двумя факторами.  
    Формат входа:
    <title>: название заболевания, связанного с этим документом
    <age>: возрастная группа
    <text>: текст документа
    Формат выхода:
    <list>
    <recom><entity1>,<entity2>,<relation></recom>, где entity1, entity2 - названия сущностей, relation - отношение между сущностями
    </list>
    Пример:
    <title>:Хронический бронхит
    <age>: взрослые
    <text>: У взрослых пациентов (>=18 лет) с легкой бронхиальной астмой рекомендуется фиксированная комбинация беклометазон+сальбутамол, для купирования симптомов и поддерживающей терапии бронхиальной астмы. Также при бронхиальной астме рекомендуется обильное питье.
    Заболевание:бронхиальная астма
    Выход:
    <list>
    <re><entity1>пациент<entity2>взрослый(>=18 лет)<relation>рекомендуется фиксированная комбинация беклометазон+сальбутамол, для купирования симптомов и поддерживающей терапии бронхиальной астмы</re>
    <
    '''
    human_prompt = '''
    <title>:{title}
    <age>:{age}
    <text>:{text}

    Выход:
    '''

    template = ChatPromptTemplate.from_messages([
        ('system',system_prompt),
        ('human',human_prompt)
    ])
    chain = template | model | StrOutputParser()

    output = chain.invoke(input={'text':text, 'title':title, 'age':age})

    return output

#------------------------------------------------------------------------------

def get_docs():
    docs = []
    with open(f'data/documents_info.json','r',encoding='utf-8') as f:
        info = json.load(f)
    for id in documents:
        docs.append(f'{id} {get_disease_title_by_id(id, info)}')
    return docs

#------------------------------------------------------------------------------
# Удаление пробелов и выделение сущностей
def get_clean_and_rec():
    with open(f'data/documents_info.json','r',encoding='utf-8') as f:
        info = json.load(f)
    for id in documents:
        with open(f'data/KR_{id}.txt','r',encoding='utf-8') as f:
            text = f.read()
        sections = split_document(text)
        full_text = '' 
        full_entity = ''
        for i in headers:
            clean_text = clean_query(model, sections.get(i))
            entity_text = entity_query(model, clean_text, get_disease_title_by_id(id, info), get_age_by_id(id, info))
            full_text += clean_text + '\n'
            full_entity += entity_text + '\n'
        with open(f'data/KR_{id}.txt','w+',encoding='utf-8') as f:
            text = f.write(full_text)
        with open(f'data/KR_{id}_rec.txt','w+',encoding='utf-8') as f:
            text = f.write(full_entity)
    return 'База знаний обновлена'

#------------------------------------------------------------------------------
# Отдельное удаление пробелов
def get_clean():
    for id in documents:
        with open(f'data/KR_{id}.txt','r',encoding='utf-8') as f:
            text = f.read()
        sections = split_document(text)
        full_text = '' 
        for i in headers:
            clean_text = clean_query(model, sections.get(i))
            full_text += clean_text + '\n'
        with open(f'data/KR_{id}.txt','w+',encoding='utf-8') as f:
            text = f.write(full_text)
    return 'Проведена очистка документов'
#------------------------------------------------------------------------------
# Отдельное выделение сущностей
def get_rec():
    with open(f'data/documents_info.json','r',encoding='utf-8') as f:
        info = json.load(f)
    for id in documents:
        with open(f'data/KR_{id}.txt','r',encoding='utf-8') as f:
            text = f.read()
        sections = split_document(text)
        full_entity = ''
        for i in headers:
            entity_text = entity_query(model, sections.get(i), get_disease_title_by_id(id, info), get_age_by_id(id, info))
            full_entity += entity_text + '\n'
        with open(f'data/KR_{id}_rec.txt','w+',encoding='utf-8') as f:
            text = f.write(full_entity)
    return 'Выделены сущности'

#------------------------------------------------------------------------------
# Функция - заглушка для проверки 
def get_todo():
    return 'Это функция заглушки'


#------------------------------------------------------------------------------
# ------------------------------------------------Для графов---------------------------------------------------------------------

def convert_entity_data(data):
    result = []
   
    node_pattern = r'<node>(.*?)</node>'

    entity1_pattern = r'<entity1>(.*?)</entity1>'
    entity2_pattern = r'<entity2>(.*?)</entity2>'
    recom_pattern = r'<recom>(.*?)</recom>'

    nodes = re.findall(node_pattern, data, re.DOTALL)
    
    for node in nodes:
        entity1 = re.search(entity1_pattern, node).group(1)
        entity2 = re.search(entity2_pattern, node).group(1)
        relation = re.search(recom_pattern, node).group(1)
        result.append((entity1, entity2, relation))
        
    return result

def draw_graph(entities):
    net = Network(height='500px', width='100%', notebook=True)
    
    # Добавляем узлы и рёбра
    for source, target, label in entities:
        net.add_node(source, label=source)
        net.add_node(target, label=target)
        net.add_edge(source, target, title=label)  # Добавляем подпись к ребру
    
    # Сохраняем граф в HTML файл
    net.save_graph('graph.html')