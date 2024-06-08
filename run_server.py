#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes


from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import os
from langchain.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_community.chat_models.yandex import ChatYandexGPT
from langchain_community.llms.llamacpp import LlamaCpp

from kg_server import run

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)



#print(chain_with_sources.invoke('Пациент: мужчина, 50 лет. Диагноз: бронхит. Степень тяжести: легкая. Дополнительно: диабет (легая степень)'))
print('init')


add_routes(
    app,
    {'patient_info':itemgetter('patient_info'),'model':itemgetter('model'), 'model_params':itemgetter('model_params'),'key':itemgetter('key')} | RunnableLambda(run),
    path="/med"
)

add_routes(
    app,
    ChatPromptTemplate.from_template('{inp}'),
    path="/test1"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=21081)