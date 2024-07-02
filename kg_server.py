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
import update_base
from graph import Graph

from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models.perplexity import ChatPerplexity
from langchain_community.chat_models.gigachat import GigaChat


def unite_docs(docs:dict):
    string = ''

    for _,text in docs.items():
        string += f'<doc> {text} </doc>\n'
    
    return string

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

def get_relevant_file(patient_info,documents,model):

    docs = unite_docs(documents)

    sys_prompt = '''
    Ты - медицинский чат бот. Твоя задача: выбрать документ с клиническими рекомендациями, который подойдет пациенту.
    Советы:
    1) Описывай свои размышления (пример: исходя из названия заболевания и возраста пациента, а также названий документов и возраста пациентов, подходящих для этих документов, следует выбрать это, так как)
    2) Если не один документ не подходит описанию пациента, в качестве индекса в ответе напиши -1.
    Формат входа:
    <disease> - название заболевания
    <patient_age> - возраст пациента
    <list>
    <doc> <id> индекс документа </id> <title> название документа </title> <age> возрастная группа </age> </doc>
    </list>
    Формат выхода:
    <thoughts> размышления </thoughts>
    <id> индекс документа </id>
    '''

    prompt = '''
    <disease> {disease}
    <patient_age> {patient_age}
    <list>
    {docs}
    </list>

    Выход:
    <thoughts>
    '''

    template = ChatPromptTemplate.from_messages([
        ('system',sys_prompt),
        ('human',prompt)
    ])

    

    chain = template | model | StrOutputParser()

    output = chain.invoke(input = {'disease': patient_info['main_diagnosis'], 'patient_age':patient_info['age'],'docs':docs})

    print('DEBUG:',output)

    output_pattern = r'<id>(.*)</id>'

    id = re.findall(output_pattern,output)[0]
    id = id.encode()

    return id

def get_graph(patient_info,model):
    with open('data/documents_info.json','r',encoding='utf-8') as f:
        documents_info = json.load(f)
    id = int(get_relevant_file(patient_info,documents_info,model))

    if (id==-1):
        return None

    with open(f'data/KR_{id}_rec.txt','r',encoding='utf-8') as f:
            text = f.read()
    entities = update_base.convert_entity_data(text)

    graph = Graph()

    for (e1,e2,re) in entities:
        graph.add_node(e1,e2,re)
    print(f'graph was found id:{id}')
    return graph

def format_nodes(nodes):
    string = ''

    for n in nodes:
        string += f'<node> {n} </node>\n'

    return string

def get_query(model, graph:Graph, patient_info):
    system = '''
    Ты медицинский чат бот.
    Твоя задача: Выбрать вершины графа знаний для получения клинических рекомендаций, релевантных описанию пациента. Для этого ты должен посмотреть на описание пациента и на набор всех вершин и выбрать те, что могут пригодиться для данного пациента.
    Советы:
    1) Используй только те названия вершин, котороые есть в графе
    Формат входа:
    <patient info> - описание пациента
    <kg nodes> <node> название вершины </node> - все вершины графа

    Пример выхода:
    <output> <node> entity1 </node> <node> entity2 </node> </output> - список вершин графа
    '''

    user = '''
    <patient info> {patient_info}

    <kg nodes> {nodes}

    Выход:
    '''

    template = ChatPromptTemplate.from_messages([
        ('system',system),
        ('human',user)
    ])

    chain = template | model | StrOutputParser()

    nodes = graph.entities_list()

    output = chain.invoke(input={'patient_info':patient_info,'nodes':format_nodes(nodes)})

    output_pattern = r'<node>\s*(.*?)\s*</node>'

    output = re.findall(output_pattern,output)

    return output 

def get_query_nodes(graph:Graph,query, max_depth = 1):
    return_nodes = set()
    
    for e in query:
        nodes = graph.find_nodes(e)
        for node in nodes:
            return_nodes.add(node)
    for _ in range(max_depth-1):
        for (e1,e2,_) in return_nodes:
            nodes = graph.find_nodes(e1)
            for node in nodes:
                return_nodes.add(node)
            nodes = graph.find_nodes(e2)
            for node in nodes:
                return_nodes.add(node)
    return return_nodes

def format_triplets(triplets):
    string = ''

    for (entity1,entity2,re) in triplets:
        string += f'<node> {entity1} - {entity2} : {re}\n'
    return string

def generate_recoms(model,triplets,patient_info):
    system = '''
    Ты медицинский чат бот.
    Твоя задача: сгенерировать набор клинических рекомендаций для пациента, основываясь на полученном наборе связей между сущностями. Из графа знаний были извлечены сущности и связи между ними, которые могут подойти для данного пациента. Некоторые из них могут быть не релевантными. Тебе необходимо выбрать те, что подходят для этого пациента и сгенерировать итоговый ответ.
    На выходе должен получиться нумерованный список.
    Советы: 
    1) Используй информацию только из полученных сущностей и связей между ними
    2) Перед ответом, распиши свои мысли (давайте подумаем, рассмотрим, какие рекомендации связаны с пациентом).
    3) Если у пациента есть заболевания, для которых нет рекомендаций в графе, игнорируй их
    4) Если ты встречаешь какие либо значениея (цифры в сущностях или в рекомендациях) старайся указывать их в итоговых рекомендациях.
    Формат входа:
    <kg> <node> сущность1 - сущность2 : связь - сущности и связи между ними, полученные из графа знаний
    <patient info> информация о пациенте
    Формат выхода:
    <thought> размышления </thought>
    <output> список рекомендаций </output>

    Пример выхода:
    <thought>Так как у пациента простуда и появился кашель, ему следует принимать препарат против кашля по 5мг 2 раза в день.</thought>
    <output> 1)Назначить препарат против кашля 5мг 2 раза в день </output>
    '''

    user = '''
    <kg> {nodes}

    <patient info> {patient_info}

    Выход:
    <thought>
    '''

    template = ChatPromptTemplate.from_messages(
        [
            ('system',system),
            ('human',user)
        ]
    )

    chain = template | model | StrOutputParser()

    output = chain.invoke(input={'nodes':triplets, 'patient_info':patient_info})

    output_pattern = r'<output>(.*)</output>'

    print('DEBUG:',output)

    output = re.findall(output_pattern,output, re.DOTALL)[0]

    return output

def run(data):

    model_name = data.get('model','')
    key = data.get('key','')
    model_params = data.get('model_params')
    temperature = model_params['temperature']
    del model_params['temperature']

    patient_info = data['patient_info']

    if('gpt' in model_name):
        os.environ['OPENAI_API_KEY'] = key
        model = ChatOpenAI(model=model_name,temperature=temperature,model_kwargs=model_params)
    elif('claude'in model_name):
        model = ChatAnthropic(model_name=model_name,api_key=key)
    elif('sonar' in model_name):
        os.environ["PPLX_API_KEY"] = key
        model = ChatPerplexity(model=model_name,temperature=temperature,model_kwargs=model_params)
    elif('local_model'==model_name):
        model = ChatOpenAI(base_url=key,api_key='llama.cpp')
    elif('GigaChat' in model_name):
        model = GigaChat(credentials=key,model_name=model_name,temperature=temperature,verify_ssl_certs=False)

    steps = 0
    print(f'[{steps}]:getting graph')
    graph = get_graph(patient_info,model)
    if(graph == None):
        print('File not founded')
        return {'status':1}

    steps+=1
    print(f'[{steps}]:getting query')
    query = get_query(model,graph,patient_info)
    print(f'query: {query}')
    steps+=1
    print(f'[{steps}]:getting nodes')
    nodes = get_query_nodes(graph,query)

    with open('graph_dump.txt','w+',encoding='utf-8') as f:
        f.write(format_triplets(nodes))

    steps+=1
    print(f'[{steps}]:generating recoms')

    patient_info_str = format_patient_info(patient_info)
    triplets = format_triplets(nodes)

    recoms = generate_recoms(model,nodes,patient_info_str)

    return_dict = {'recoms':recoms,'triplets':triplets, 'status':0}

    print(return_dict)
    
    return return_dict