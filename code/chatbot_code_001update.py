#  This chatbot Python Flask code is to use GPT API in Azure: no datasources 
#  python chatbot.py
#  http://127.0.0.1:5000/ (Press CTRL+C to quit)
#  This is the new version Python Flask based on OpenAI version > 1.0 and 
#  some update based on the previous Flask code: chatbot_code_001.py
#  I also update the chat.html 

import os
from flask import Flask, request, jsonify, render_template
import openai
from openai import AzureOpenAI as AzureOpenAI0
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
openai.api_version = "2023-08-01-preview"
# Azure OpenAI setup
openai.api_base = "https://*************"
openai.api_key = "**************"
deployment_id = "gpt4"

# For OpenAI ver > 1.0
client = AzureOpenAI0(
  api_key = openai.api_key,  
  api_version = openai.api_version,
  azure_endpoint = openai.api_base
)

app = Flask(__name__, template_folder='templates')
app.secret_key = "super secret key"
app.config['TEMPLATES_AUTO_RELOAD'] = True
img_folder = os.path.join('static', 'img')
app.config['UPLOAD_FOLDER'] = img_folder

with open('creditcard_QA.txt', 'r', encoding='utf-8') as file:
    lines = file.read().split('\n\n')  # Split the content by empty lines

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
    os.environ["AWS_ACCESS_KEY_ID"] = "*************"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "******************"

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

    request = str(find_text_filter) 

    return request

# Define the database connection string
db_connection_string = "postgresql://postgres:***********"
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

# This is new function for removing the chat history for the first visiting ChatBot
def reset_history_internal():
    try:
        conn = sqlite3.connect('conversation_history.db')
        c = conn.cursor()
        c.execute('DELETE FROM conversation_history')
        c.execute('DROP TABLE IF EXISTS conversation_history')
        conn.commit()
        return True
    except Exception as e:
        print(str(e))
        return False
      
# This is new function for removing the chat history for the first visiting ChatBot        
@app.route('/reset_history', methods=['GET'])
def reset_history():
    if reset_history_internal():
        return jsonify({'success': 'Conversation history reset successfully'})
    else:
        return jsonify({'error': 'Failed to reset conversation history'})
    
@app.route('/')
def chat():
    if not reset_history_internal():
        return jsonify({'error': 'Failed to reset conversation history'})
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

# There are some update on this function: 
# 1) OpenAI API's response 2) arrangement of prompt: message_lst 3) save dialogs into DB  
@app.route('/send', methods=['POST'])
def send_message():
    user_message = request.json['message']
    user_request = str(get_request(user_message))

    # Retrieve the conversation history
    user_id = 'unique_user_id'  # You should replace this with the actual user identifier
    session_id = 'unique_session_id'  # You should replace this with the actual session identifier

    notice = '. Note, do not randomly fabricate unless you more than 80% know, \
             otherwise say I do not know, also your answer should be clear, brief, \
             do not repeat the question '
    
    your_knowledge = ' Use the general knowledge of credit card business, '
                      
    conversation_history = get_conversation_history(user_id, session_id)
    
    history_message = ';'.join(conversation_history) 

    start_messages = [{"role": "system", "content": "You are a good helper assistant for anwsering credit card business"},
          {"role": "user",   "content": notice},
          {"role": "assistant", "content": " sure, I will follow this instruction "},
          {"role": "user", "content": " the following questions from me and the answers from you are the knowledge review or rehearsal in order to answer user's real question "},
          {"role": "assistant", "content": " sure, I will use the questions from you and the answers from me as the knowledge review or rehearsal "}
         ]
          
    rules_messages_st = [{"role": "user",   "content": "let me start to ask the first question and you will answer my question for knowledge review or rehearsal purpose until I say: knowledge review completes. "},
                          {"role": "assistant", "content": " sure, I will answer the questions as the knowledge review or rehearsal until you say: knowledge review completes."}]     
     
    rules_messages_ed = [{"role": "user",   "content": "knowledge review completes."},
                         {"role": "assistant", "content": " sure, only use the questions and answers above as knowledge review, they are not real questions from user"}]     
         
    history = [{"role": "user",   "content": 'Here is our conversation history including the questions from user and anwsers from assistant, please do not answer them, only use the information to answer the most recent question ' + history_message},
              {"role": "assistant", "content": "Yes, I will use the conversation history information to keep chatting"}]
            
    end_messages = [{"role": "user", "content": 'Now, ' + your_knowledge + ' please answer the real question from user: ' + user_request + ', \
           If there are more than two consecutive X letters \
           in the question text, then your answer should be added with the following \
           annotation: I have filtered out your personal privacy information before answering your question. '}]

    if len(conversation_history)>=2:
        message_lst = start_messages + rules_messages_st + rules + \
                      rules_messages_ed + history + end_messages    
    else:                  
        message_lst = start_messages + rules_messages_st + rules + \
                      rules_messages_ed + end_messages    
    
    print ("**********history_message**************:", history_message)
        
    response_final = client.chat.completions.create(
        model = deployment_id,
        temperature=0.25,
        max_tokens = 180,
        messages = message_lst
    )
    
    bot_response_final = response_final.choices[0].message.content
    bot_response_final = bot_response_final.strip()
    bot_response_final = bot_response_final.replace('Answer:', "")

    # Save the user's question and bot's answer to the database
    # Add the current user message to the conversation history
    add_message_to_history(user_id, session_id, ' Question from user: ' + user_request)
    add_message_to_history(user_id, session_id, 'Answer from assistant: ' + bot_response_final)

    dialog_table_entry = {'question': user_message, 'answer': bot_response_final}
    session.execute(dialog_table.insert().values(dialog_table_entry))
    session.commit()

    return jsonify({'response': bot_response_final})

if __name__ == '__main__':
    app.run()

#################
