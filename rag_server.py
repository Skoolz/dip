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


from langchain_anthropic import ChatAnthropic


def init_db():
    global db
    db = Chroma(persist_directory="./chroma_db",embedding_function= OpenAIEmbeddings(model='text-embedding-3-large'))


def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])


def generate_queries(model):

    sys_template = 'Ты - медицинский чат бот. Твоя задача: помочь врачу в установлении диагноза и выбора лечения.'
    prompt = '''
    Клиническая рекомендация содержит пункты: диагностика, лечение, профилактика, жалобы и анамнез и другое.
    Необходимо сгенерировать несколько текстовых запросов к базе клинических рекомендаций исходя из оригинального запроса.
    Текстовый запрос должен быть ориентирован на один из этих пунктов. Для каждого запроса выбери необходимую для него часть оригинального запроса.

    Пример: запрос - \"Пациент страдает хроническим бронхитом. Также у пациента есть астма\".
    Возможный вариант запроса: \"Лечение хронического бронхита при наличии астмы\"

    Всего необходимо сгенировать максимум 4 запроса. Каждый запрос начинается с новой строки
    '''

    prompt2 = '''
    Запрос:{original_query}
    Вывод:
    '''

    chat_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(sys_template),
        ('human',prompt),
        ('human',prompt2)
    ]
    )

    output = chat_template | model | StrOutputParser() | (lambda x: x.split('\n'))

    return output


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        loads(doc)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

def unite_docs(docs):
    return '\n---\n'.join(docs.values())

def get_id(raw_ids):
    match = re.search(r'<id>(.*?)</id>', raw_ids)
    result = None
    if match:
        result = match.group(1)
    return result


def get_relevant_file(disease,model):

    with open('data/documents_info.json','r',encoding='utf-8') as f:
        documents_info = json.load(f)

    sys_prompt = 'Ты - медицинский чат бот. Твоя задача: помочь врачу в установлении диагноза и выбора лечения.'

    prompt_1 = '''
    Исходя из назвний документов необходимо выбрать один документ, подходящий для данного случая.
    На вход подается список документов и связанных с ними заболеваниями в виде: \"<id>(индекс документа)</id>:\n<title>(Название документа)</title>\n---\", а также название болезни.
    Необходимо выбрать такой документ, который больше всего по смыслу подходит к заболеванию.
    На выходе необходимо написать индекс документа в формате \"<id>(индекс документа)</id>\"

    Пример:
    Документы:
    <id>4</id>:
    <name>Бронхит</name>
    ---
    <id>25</id>:
    <name>Астма</name>
    ---

    Заболевание:
    Бронхиальная астма

    Вывод:
    <id>25</id>
    '''

    prompt_2 = '''
    Документы:
    {docs}
    Заболевание:
    {disease}
    Вывод:
    '''

    template = ChatPromptTemplate.from_messages([
        ('system',sys_prompt),
        ('human',prompt_1),
        ('human',prompt_2)
    ])

    

    chain = template | model | StrOutputParser()

    raw_id = chain.invoke(input={'docs':unite_docs(documents_info),'disease':disease})

    return get_id(raw_id)

def extract_relationships(text):
    # Шаблон для поиска подстрок в формате <re><entity1>...<entity2>...<relation>...</re>
    pattern = r'<re><entity1>(.*?)<entity2>(.*?)<relation>(.*?)</re>'
    
    # Используем findall для поиска всех совпадений с шаблоном в тексте
    matches = re.findall(pattern, text)
    
    # Преобразуем каждое совпадение в словарь
    relationships = [{'entity1': match[0], 'entity2': match[1], 'relation': match[2]} for match in matches]
    
    return relationships

def convert_relationships_to_str(relationships):
    return "Рекомендация:"+'\nРекомендация:'.join([f"{r['entity1']},{r['entity2']}:{r['relation']}" for r in relationships])

def generate_re(text,disease,model):
    system_prompt = '''
    Ты - медицинский чатбот. Твоя задача - найти в данном тексте рекомендации в виде сущностей и отношений между ними. Обычно это условия, страдает ли пациент какими то дополнительными заболевания, курит или пьет, взрослый или нет и так далее.
    На выходе - список сущностей и связей между ними. Формат вывода:
    <list>
    <re><entity1>название сущнсоти<entity2>название сущнсоти<relation>взаимосвязь между ними</re>
    </list>
    Каждое найденное отношение начинается с новой строки
    На вход подается текст и название заболевания. Его, как сущность разспознавать не надо. Ты должен найти зависимости между дополнительными факторами и пациентом. Если дана общая рекомендция, то ставь вместо entity1 название заболевания, а вместо entity2 прочерк.
    Пример:
    Текст: У взрослых пациентов (>=18 лет) с легкой бронхиальной астмой рекомендуется фиксированная комбинация беклометазон+сальбутамол, для купирования симптомов и поддерживающей терапии бронхиальной астмы. Также при бронхиальной астме рекомендуется обильное питье.
    Заболевание:бронхиальная астма
    Вывод:
    <list>
    <re><entity1>пациент<entity2>взрослый(>=18 лет)<relation>рекомендуется фиксированная комбинация беклометазон+сальбутамол, для купирования симптомов и поддерживающей терапии бронхиальной астмы</re>
    <re><entity1>бронхиальная астма<entity2>-<relation>рекомендуется обильное питье
    </list>
    '''

    human_prompt = '''
    Текст:{text}
    Заболевание:{disease}

    Вывод:
    '''

    template = ChatPromptTemplate.from_messages([
        ('system',system_prompt),
        ('human',human_prompt)
    ])


    chain = template|model|StrOutputParser()
    relationships = chain.invoke(input={'text':text,'disease':disease})
    relationships = extract_relationships(relationships)
    relationships = convert_relationships_to_str(relationships)

    return relationships

def process(disease,text, documents_per_query=3, k=5,model=None):
    print('Getting document id')
    doc_id = get_relevant_file(disease,model)
    print('Document id:',doc_id)
    retriever = db.as_retriever(search_kwargs={"k": documents_per_query,'filter':{'source':{'$in':[f'data\\KR_{doc_id}.txt']}}})

    print('Getting document chunks')
    documents = (generate_queries(model) | retriever.map() | reciprocal_rank_fusion).invoke(text)
    documents = documents[:k]
    print('Documents count:',len(documents))

    sys_template = '''
    Ты - медицинский чат бот. 
    Твоя задача: сгенерировать набор клинических рекомендаций для данного пациента, основываясь на данном контексте. 
    Вывод должен выглядеть в виде нумерованного списка. 
    Формат вывода \"<recom_list>(Список)</recom_list>\". 
    Каждая рекомендация должна ссылаться на описание пациента (к примеру \"Если у пациента пневмония и аллергия на на N, то ему следует (текст рекомендации))\")
    Советы:
    1) В документе может отсутствовать информация по некоторым частям описанию пациента. Не нужно писать рекомендации для этих частей, просто игнорируй их.
    2) Старайся делать рекомендации по основному диагнозу.
    3) Если пациент болеет чем то еще, ищи в документе, связан ли основной диагноз с этим. Если нет, то рекомендации по побочным заболеваниям писать не надо.
    Запреты:
    1) Запрещены рекомендации, текст которых не содержит информации из документа! Вся информация в рекомендации должна быть взята из документа.
    
    Пример:
    (часть документа):
    При пневмонии и ЧСС>90, следует сделать рентген грудной клетки

    (часть информации о пациенте):
    ЧСС:95
    Основной диагноз: пневмония

    Вывод:
    <recom_list>
    1) Так как пациент у пациента пневмония и ЧСС > 90 (95), следует сделать рентген грудной клетки
    </recom_list>
    '''

    template = '''
    Документ:
    {context}

    Информация о пациенте:{question}

    Вывод (соблюдай формат вывода):
    '''


    prompt = ChatPromptTemplate.from_messages(
    [SystemMessagePromptTemplate.from_template(sys_template),HumanMessagePromptTemplate.from_template(template)])

    #print('Extracting recommendations')
    #document_text = format_docs(documents)
    #recommendations = generate_re(document_text,disease)

    chain = (
        {'context':itemgetter('context')|RunnableLambda(format_docs),'question':itemgetter('query'),'recoms':itemgetter('recoms')}
        | prompt
        | model
        | StrOutputParser()
    )

    chain_with_sources = RunnablePassthrough().assign(answer=chain)
    print('Final query')
    return chain_with_sources.invoke(input={'query':text,'context':documents,'recoms':''})

def run(data):

    model_name = data.get('model','')
    key = data.get('key','')
    open_ai_key = data.get('open_ai','')

    os.environ['OPENAI_API_KEY'] = open_ai_key

    init_db()
    print('key:',key)

    if('gpt' in model_name):
        model = ChatOpenAI(model=model_name)
    elif('claude'in model_name):
        model = ChatAnthropic(model_name=model_name,api_key=key)
    result = process(disease=data['disease'],text=data['patient_info'],model=model)
    return result