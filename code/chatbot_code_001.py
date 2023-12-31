# this chatbot Python Flask code is to use GPT API in Azure: no datasources 
#  python chatbot.py
#  http://127.0.0.1:5000/ (Press CTRL+C to quit)

import os
from flask import Flask, request, jsonify, render_template
import openai
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, text, DateTime, Date, Table, MetaData, inspect, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from langchain_experimental.comprehend_moderation import AmazonComprehendModerationChain
import boto3
from langchain.llms.fake import FakeListLLM
from langchain.prompts import PromptTemplate
from langchain_experimental.comprehend_moderation.base_moderation_exceptions import ( ModerationPiiError,)
import sqlite3

wholepath = r'C:\chatbot'
os.chdir(wholepath)

openai.api_type = "azure"
# Azure OpenAI on your own data is only supported by the 2023-08-01-preview API version
openai.api_version = "2023-07-01-preview"
# Azure OpenAI setup
openai.api_base = "https://creditcardchatbot-gpt4.openai.azure.com/"
openai.api_key = "******************"
deployment_id = "gpt4"


app = Flask(__name__, template_folder='templates')
app.secret_key = "super secret key"
app.config['TEMPLATES_AUTO_RELOAD'] = True
img_folder = os.path.join('static', 'img')
app.config['UPLOAD_FOLDER'] = img_folder

with open('creditcard_QA.txt', 'r', encoding='utf-8') as file:
    lines = file.read().split('\n\n')

questions = []
answers = []

for pair in lines:
    if pair.startswith("Q:"):
        question = pair.split('\n')[0][2:].strip()
        answer = pair.split('\n')[1][2:].strip()
        questions.append(question)
        answers.append(answer)

data = {'question': questions, 'answer': answers}
df = pd.DataFrame(data)

rules = []
for index, row in df.iterrows():
    user_dict = {'role': 'user', 'content': row['question']}
    rules.append(user_dict)
    assistant_dict = {'role': 'assistant', 'content': row['answer']}
    rules.append(assistant_dict)


def langchin_filter(request_str):
    os.environ["AWS_ACCESS_KEY_ID"] = "***"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "***********"

    comprehend_client = boto3.client("comprehend", region_name="us-east-1")

    comprehend_moderation = AmazonComprehendModerationChain(
    client=comprehend_client,
    verbose=True,  
    )

    from langchain_experimental.comprehend_moderation import (
        BaseModerationConfig,
        ModerationPiiConfig,
        ModerationPromptSafetyConfig,
        ModerationToxicityConfig,
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

    notice = '. Note, do not randomly fabricate unless you more than 75% know, \
             otherwise say I do not know, also your answer should be clear, brief, \
             do not repeat the question '

    request = ' Combine with general knowledge of credit card business, \
          please answer the following question: ' + str(find_text_filter) + notice

    return request

# Define the database connection string
db_connection_string = "postgresql://postgres:***************"
# Create the SQLAlchemy engine
engine = create_engine(db_connection_string)
# Declare a base for the declarative class
Base = declarative_base()
# Define the message_hist table separately
dialog_table = Table(
    'message_hist', Base.metadata,
    Column('id', Integer, primary_key=True),
    Column('question', String),
    Column('answer', String),
    Column('timestamp', Date, default=datetime.utcnow().date)
)

# Create the table in the database only if it doesn't exist
inspector = inspect(engine)
if not inspector.has_table('message_hist'):
    Base.metadata.tables['message_hist'].create(bind=engine)

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session()

def query_dialog_function(start_date, end_date):
    # Convert start_date and end_date strings to datetime objects
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

@app.route('/reset_history', methods=['GET'])
def reset_history():
    try:
        # Delete the temporary table 'conversation_history'
        conn = sqlite3.connect('conversation_history.db')
        c = conn.cursor()
        c.execute('DROP TABLE IF EXISTS conversation_history')
        conn.commit()

        return jsonify({'success': 'Conversation history reset successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})
    
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
    
    # Construct the conversation history for OpenAI API
    conversation_history = [{"role": "user", "content": 'I will furnish you with the ongoing conversation history to ensure that our interaction remains cohesive within this session, and only answer the final question'}] + \
                                  [{"role": "assistant", "content": "Yes, I will use the previous information and only answer the most recent question"}] + \
                                  [{"role": "user", "content": 'conversation history, do not answer, just use them to answer the most recent question ' + msg} for msg in conversation_history] + \
                                  [{"role": "assistant", "content": "Yes, I will use the previous information"}] 

    end_messages = [{"role": "user", "content": 'Now answer my question: ' + user_request + ', please provide the answer using the full\
           knowledge in our conversational history,  If there are more than two consecutive X letters \
           in the question text, then your answer should be added with the following \
           annotation: I have filtered out your personal privacy information before answering your question. '}]

    message_lst = start_messages + rules + conversation_history + end_messages    

    response_final = openai.ChatCompletion.create(
        deployment_id=deployment_id,
        temperature=0.25,
        max_tokens = 180,
        messages= message_lst
    )

    bot_response_final = response_final['choices'][0]['message']['content']
    bot_response_final = bot_response_final.strip()
    bot_response_final = bot_response_final.replace('Answer:', "")

    add_message_to_history(user_id, session_id, bot_response_final)

    dialog_table_entry = {'question': user_message, 'answer': bot_response_final}
    session.execute(dialog_table.insert().values(dialog_table_entry))
    session.commit()

    return jsonify({'response': bot_response_final})

if __name__ == '__main__':
    app.run()


