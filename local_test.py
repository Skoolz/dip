from openai import OpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# client = OpenAI(
#     base_url="http://149.36.0.216:40205/v1",
#     api_key="llama.cpp"
# )

model = ChatOpenAI(base_url='http://65.109.73.2:7548/v1',api_key="llama.cpp")

# messages = [
#     {'role':'system','content':'You are helpful assistant'},
#     {'role':'user','content':'Write function in python to find normal of numpy array'}
# ]



# output = client.chat.completions.create(messages=messages,model=model, stream=True, temperature=0.2)

# text = ''
# for chunk in output:
#     text+=f'{chunk.choices[0].delta.content}\n'

# print(text)

messages = [
    ('system','You are helpful assistant'),
    ('human','Что делать при высокой температуре?')
]

template = ChatPromptTemplate.from_messages(messages)

chain = template | model | StrOutputParser()

print(chain.invoke(input={}))