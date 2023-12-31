# this chatbot Python Flask code is to use GPT API in Azure: datasources 
# python chattry.py
# http://127.0.0.1:5000/ (Press CTRL+C to quit)

import openai, os, requests
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, text, DateTime, Date, Table, MetaData, inspect, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from langchain_experimental.comprehend_moderation import AmazonComprehendModerationChain
import boto3
from langchain.llms.fake import FakeListLLM
from langchain.prompts import PromptTemplate
from langchain_experimental.comprehend_moderation.base_moderation_exceptions import ( ModerationPiiError,)
from datetime import datetime, timedelta

wholepath = r'C:\chatbot'
os.chdir(wholepath)

openai.api_type = "azure"
openai.api_version = "2023-07-01-preview"
openai.api_base = "https://creditcardchatbot-gpt4.openai.azure.com/"
openai.api_key = "****************"

deployment_id = "gpt432k"
search_endpoint = "https://creditcardchatbot.search.windows.net"
search_key = "******************************"
search_index_name = "creditcardrules"
chat_completion_temperature = 0.2

def get_conversation_history(user_id, session_id):
    conn = sqlite3.connect('conversation_history.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS conversation_history (user_id TEXT, session_id TEXT, message TEXT)')
    conn.commit()

    conversation_history = []
    for row in c.execute('SELECT message FROM conversation_history WHERE user_id = ? AND session_id = ?', (user_id, session_id)):
        conversation_history.append(row[0])

    return conversation_history

def add_message_to_history(user_id, session_id, message):
    conn = sqlite3.connect('conversation_history.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS conversation_history (user_id TEXT, session_id TEXT, message TEXT)')
    c.execute('INSERT INTO conversation_history VALUES (?, ?, ?)', (user_id, session_id, message))
    conn.commit()

def setup_byod(deployment_id: str) -> None:
    class BringYourOwnDataAdapter(requests.adapters.HTTPAdapter):

        def send(self, request, **kwargs):
            request.url = f"{openai.api_base}/openai/deployments/{deployment_id}/extensions/chat/completions?api-version={openai.api_version}"
            return super().send(request, **kwargs)

    session = requests.Session()

    # Mount a custom adapter which will use the extensions endpoint for any call using the given `deployment_id`
    session.mount(
        prefix=f"{openai.api_base}/openai/deployments/{deployment_id}",
        adapter=BringYourOwnDataAdapter()
    )

    openai.requestssession = session

setup_byod(deployment_id)

app = Flask(__name__, template_folder='templates')
app.secret_key = "super secret key"
app.config['TEMPLATES_AUTO_RELOAD'] = True
img_folder = os.path.join('static', 'img')
app.config['UPLOAD_FOLDER'] = img_folder

def langchin_filter(request_str):
    os.environ["AWS_ACCESS_KEY_ID"] = "*********"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "***************"

    comprehend_client = boto3.client("comprehend", region_name="us-east-1")

    comprehend_moderation = AmazonComprehendModerationChain(
    client=comprehend_client,
    verbose=True,  
    )

    from langchain_experimental.comprehend_moderation import (
        BaseModerationConfig,
        ModerationPiiConfig,
    )

    pii_labels = ["SSN", "DRIVER_ID", "ADDRESS",'EMAIL','PHONE', 'NAME']
    pii_config = ModerationPiiConfig(labels=pii_labels, redact=True, mask_character="X")
    
    moderation_config = BaseModerationConfig(
        filters=[pii_config]
    )

    comp_moderation_with_config = AmazonComprehendModerationChain(
        moderation_config=moderation_config,  # specify the configuration
        client=comprehend_client,  # optionally pass the Boto3 Client
        verbose=True,
    )

    template = """Question: {question}

    Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    responses = [
        "Final Answer: " + request_str,
    ]

    llm = FakeListLLM(responses=responses)

    chain = (
        prompt
        | comp_moderation_with_config
        | {"input": (lambda x: x["output"]) | llm}
        | comp_moderation_with_config
    )

    Q = request_str
    try:
        response = chain.invoke(
            {
            "question": "provide me the similar information:"
            }
        )
    except Exception as e:
        Q = request_str
        print('error.......')
    else:
        Q = response["output"][13:]
        print(Q)

    return Q       

def get_request(find_text):
    find_text_filter = langchin_filter(find_text)
    notice = '. Note, do not randomly fabricate unless you more than 40% know, \
             otherwise say I do not know, also your answer should be clear, brief, \
             do not repeat the question '

    request = ' Combine with general knowledge of help desk of credit card business, \
          please answer the following question: ' + str(find_text_filter) + notice

    return request

db_connection_string = "postgresql://postgres:*****************"
engine = create_engine(db_connection_string)
Base = declarative_base()
dialog_table = Table(
    'message_hist', Base.metadata,
    Column('id', Integer, primary_key=True),
    Column('question', String),
    Column('answer', String),
    Column('timestamp', Date, default=datetime.utcnow().date)
)

inspector = inspect(engine)
if not inspector.has_table('message_hist'):
    Base.metadata.tables['message_hist'].create(bind=engine)

Session = sessionmaker(bind=engine)
session = Session()

def query_dialog_function(start_date, end_date):
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")

    Session = sessionmaker(bind=engine)
    session = Session()
    query = text(
        "SELECT question, answer FROM message_hist "
        "WHERE timestamp >= :start_datetime AND timestamp <= :end_datetime"
    )
    result = session.execute(query, {"start_datetime": start_datetime, "end_datetime": end_datetime})
    result_panels = [{"question": row["question"], "answer": row["answer"]} for row in result]
    session.close()
    return result_panels

@app.route('/')
def chat():
    return render_template('chat.html')

@app.route('/dialog_hist')
def show_dialog():
    return render_template('query_dialog.html')

@app.route('/query_dialogby', methods=['GET'])
def query_dialogby():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    if start_date is None or end_date is None:
        return jsonify({'error': 'Missing start_date or end_date parameter'})
    try:

        result_panels = query_dialog_function(start_date, end_date)

        return jsonify(result_panels)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/send', methods=['POST'])
def send_message():
    user_message = request.json['message']
    user_request = str(get_request(user_message))

    user_id = 'unique_user_id'  # You should replace this with the actual user identifier
    session_id = 'unique_session_id'  # You should replace this with the actual session identifier
    conversation_history = get_conversation_history(user_id, session_id)

    # Add the current user message to the conversation history
    add_message_to_history(user_id, session_id, user_request)

    start_messages = [{"role": "system", "content": "You are a good helper assistant for anwsering credit card business"},
          {"role": "user",   "content": "Can you answer some questions on behalf expert "},
          {"role": "assistant", "content": " sure, I can "}]
    

    conversation_history = [{"role": "user", "content": 'I will furnish you with the ongoing conversation history to ensure that our interaction remains cohesive within this session, and only answer the final question'}] + \
                                  [{"role": "assistant", "content": "Yes, I will use the previous information and only answer the most recent question"}] + \
                                  [{"role": "user", "content": 'conversation history, do not answer, just use them to answer the most recent question ' + msg} for msg in conversation_history] + \
                                  [{"role": "assistant", "content": "Yes, I will use the previous information"}] 

    end_messages = [{"role": "user", "content": 'Now answer my question: ' + user_request + ', please provide the answer using the full\
           knowledge in our conversational history,  If there are more than two consecutive X letters \
           in the question text, then your answer should be added with the following \
           annotation: I have filtered out your personal privacy information before answering your question. '}]

    message_lst = start_messages + conversation_history + end_messages    
    
    response_final = openai.ChatCompletion.create(
    messages = message_lst,
    deployment_id = deployment_id,
    temperature=chat_completion_temperature,
    dataSources=[
        {
            "type": "AzureCognitiveSearch",
            "parameters": {
                "endpoint": search_endpoint,
                "key": search_key,
                "indexName": search_index_name,
                "inScope": False,
            }
        }
    ]
    )

    bot_response_final = response_final['choices'][0]['message']['content']
    bot_response_final = bot_response_final.strip()
    bot_response_final = bot_response_final.replace('Answer:', "")

    dialog_table_entry = {'question': user_message, 'answer': bot_response_final}
    session.execute(dialog_table.insert().values(dialog_table_entry))
    session.commit()
    return jsonify({'response': bot_response_final})

if __name__ == '__main__':
    app.run()




