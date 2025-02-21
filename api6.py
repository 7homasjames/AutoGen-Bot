from fastapi import FastAPI
from dotenv import load_dotenv
import os
import hashlib
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from semantic_kernel.functions import KernelArguments
import chromadb
import json

import sqlite3
import openai
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings, OpenAITextEmbedding
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin

from autogen import AssistantAgent, UserProxyAgent
import openai


# Initialize Semantic Kernel
kernel = Kernel()

# Prompt Template for Chat Completion with Grounding
prompt_template = """
    You are a chatbot that can have a conversation about any topic related to the provided context.
    Give explicit answers from the provided context or say 'I don't know' if it does not have an answer.
    Provided context: {{$db_record}}

    User: {{$query_term}}
    Chatbot:"""

os.getenv("GLOBAL_LLM_SERVICE") == "OpenAI"

# Add OpenAI Chat Completion Service
openai_service = OpenAIChatCompletion(
    api_key=os.getenv("OPENAI_API_KEY"),
    ai_model_id="gpt-3.5-turbo"
)
kernel.add_service(openai_service)

chat_execution_settings = OpenAIChatPromptExecutionSettings(
    ai_model_id="gpt-3.5-turbo",
    max_tokens=1000,
    temperature=0.0,
    top_p=0.5
)


chat_prompt_template_config = PromptTemplateConfig(
    template=prompt_template,
    name="grounded_response",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="db_record", description="The database record", is_required=True),
        InputVariable(name="query_term", description="The user input", is_required=True),
    ],
    execution_settings=chat_execution_settings,
)

chat_function = kernel.add_function(
        function_name="ChatGPTFunc",
        plugin_name="chatGPTPlugin",
        prompt_template_config=chat_prompt_template_config
        )


# Configure environment variables
load_dotenv()

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Create or get collection
#collection = chroma_client.get_or_create_collection(name="my_collection")
#print("Collection:", collection)

chroma_client = chromadb.Client()

# Create separate collections for CSV and PDF
csv_collection = chroma_client.get_or_create_collection(name="csv_collection")
pdf_collection = chroma_client.get_or_create_collection(name="pdf_collection")



# Function to generate a unique hash for a given text
def generate_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

# Function to upsert documents into the appropriate collection
def upsert_documents(documents, file_type):
    # Choose the collection based on file type
    collection = csv_collection if file_type == "csv" else pdf_collection
    #print("Collection:", documents)
    # Prepare documents and metadata for ChromaDB
    ids = [generate_hash(doc['line']) for doc in documents]
    texts = [doc['line'] for doc in documents]
    #print("Texts:", texts)
    embeddings = [doc['embedding'] for doc in documents]

    # Upsert into ChromaDB
    collection.upsert(documents=texts, ids=ids, embeddings=embeddings)
    return texts

# Function to query the appropriate collection
def query_collection(query_texts, n_results, file_type):
    # Choose the collection based on file type
    #print("File Type:",file_type)
    collection = csv_collection if file_type == "csv" else pdf_collection
    all_docs = collection.get()
    #print("Stored Documents:", all_docs)

    try:
        results = collection.query(
            query_texts=query_texts,
            n_results=n_results,
            include=["documents", "embeddings"]
        )
        #print("Query Results:", results)
        return results
    except Exception as e:
        return None




# Load the model
model = SentenceTransformer(os.getenv("MODEL_NAME"))

app = FastAPI()

# Data models
class Item(BaseModel):
    id: str
    line: str
    filename: str
    page_number: str = "1"  # Default to "1" if missing

class Docs(BaseModel):
    items: List[Item]


class Query(BaseModel):
    query: str

class QA(BaseModel):
    query: str
    context: str

class SQLQueryRequest(BaseModel):
    question: str
    db_path: str


# AutoGen Configuration
llm_config = {
    "model": "gpt-3.5-turbo",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0.3
}

# Initialize Agents
assistant = AssistantAgent(name="RAGAssistant", llm_config=llm_config)
user_proxy = UserProxyAgent(name="UserProxy", human_input_mode="NEVER")

@app.post("/autogen_response/")
async def autogen_response(item: QA):
    query = item.query
    context = item.context

    # Store chat history
    chat_history = []

    # Define response callback function
    def response_callback(message):
        if isinstance(message, dict):
            chat_history.append({
                "role": message.get("role", "assistant"),
                "name": message.get("name", "RAGAssistant"),
                "content": message.get("content", "No response generated.")
            })
        return message.get("content", "No response generated.")

    # Explicitly add user's message to chat history
    chat_history.append({
        "role": "user",
        "name": "UserProxy",
        "content": f"User Query: {query}\nContext: {context}"
    })

    # Start chat interaction
    response_content = user_proxy.initiate_chat(
        assistant,
        message=f"User Query: {query}\nContext: {context}\nGenerate a precise response.",
        response_callback=response_callback
    )

    # Ensure response content is properly captured
    if isinstance(response_content, dict):
        response_content = response_content.get("content", "No response generated.")

    # Debugging: Print response & chat history
    print("Final Response:", type(response_content))
    print(dir(response_content))
    print(type(response_content.__dict__))

    #print("Chat History:", chat_history)

    return {
        "output": response_content.__dict__ if response_content else "No response generated."
    }



@app.post("/sql_query/")
async def generate_sql_query_and_fetch_result(request: SQLQueryRequest):
    context = request.db_path
    que=request.question

    # messages = [
    #         {"role": "system", "content": "You are an expert in SQL."},
    #         {"role": "user", "content": f"Generate a valid SQL query for the following question. Do not include explanations, comments, or additional text.\n\nContext:\n{context}\n\nQuestion:\n{que}"}
    #     ]

    messages = [
        {"role": "system", "content": "You are an expert in SQL and must generate correct and syntactically valid SQL queries."},
        {"role": "user", "content": f"""
            Generate a valid SQL query for the following question. 
            Do not include explanations, comments, or additional text.

            **Rules:**
            - Extract the table name from the given context.
            - If the database name is 'breast_cancer_db.sqlite', assume the table name is 'breast_cancer_data'.
            - Do **not** use 'COUNT(DISTINCT *)' as it is invalid in SQL.
            - When counting distinct rows, use `COUNT(*)` on a subquery like `SELECT COUNT(*) FROM (SELECT DISTINCT * FROM table_name) AS unique_rows;`
            - Always use correct SQL syntax based on standard SQL rules.
            - When using `AVG()`, ensure that it's used to average over groups after summation is performed, not within a subquery sum.

            **Context:**
            {context}

            **Question:**
            {que}
        """}
    ]


  
    client = openai.AsyncOpenAI()  # Corrected for OpenAI >=1.0.0
    response = await client.chat.completions.create(  # Corrected method
            model="gpt-4",  # Use "gpt-4" or "gpt-3.5-turbo"
            messages=messages,
            max_tokens=200,
            temperature=0
        )

    sql_query = response.choices[0].message.content.strip()
    print("SQL Query:", sql_query)
  
    # Execute the SQL query in SQLite
    conn = sqlite3.connect(request.db_path)
    #print("Done connection")
    cursor = conn.cursor()
    #print("Done cursor")
    
    cursor.execute(sql_query)
    #print("Done execute")
    rows = cursor.fetchall()
    #print("Rows:", rows)
    column_names = [description[0] for description in cursor.description]
    #print("Column Names:", column_names)

    conn.close()

    # Format results
    result_table = f"SQL Query:\n{sql_query}\nResults:\n"
    #print("result_table:", result_table)
    if rows:
        result_table += f"{' | '.join(column_names)}\n"
        result_table += "\n".join([" | ".join(map(str, row)) for row in rows])
    else:
        result_table += "No results found."


    return {"sql_query": sql_query, "result": result_table}




@app.post("/push_docs/")
async def push_docs(item: Docs):
    try:
        docs = item.dict()["items"]
        #print("Docs:", docs)
        file_type = docs[0]["filename"].split(".")[-1].lower()  # Extract file type (csv or pdf)
        #print("File Type:", file_type)

        # Add embeddings
        for doc in docs:
            doc['embedding'] = model.encode(doc['line']).tolist()
        #print("Docs with Embeddings:", docs)

        # Upsert into the appropriate collection
        ids = upsert_documents(docs, file_type)
        #print("Upserted IDs:", ids)
        return {"inserted_ids": ids, "file_type": file_type}
    except Exception as e:
        return {"error": str(e)}


@app.post("/context/")
async def context(item: Query):
    try:
        query = item.query
        file_type = item.query.split(":")[0].lower()  # File type passed as a prefix (e.g., "csv:question")

        #print("File Type:", file_type)
        query_embedding = model.encode(query.split(":")[1]).tolist()  # Extract the actual query

        # Query the appropriate collection
        results = query_collection(query_texts=[query.split(":")[1]], n_results=5, file_type=file_type)
        #print("Results_api:", results)

        resp = {
            "docs": [],
            "filenames": [],
            "page_number": []
        }

        for result_group in results["documents"]:
            for result in result_group:
                try:
                    document_data = json.loads(result)
                    resp["docs"].append(document_data)
                    #print("Document Data:", document_data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding document: {result}, Error: {e}")

        return resp
    except Exception as e:
        return {"error": str(e)}

@app.post("/response/")
async def response(item: QA):
    try:
        print("Item",item)
        query = item.query
        context = item.context
        print("Query:", query)
        print("Context:", context)
        arguments = KernelArguments(db_record=context, query_term=query)

        result = await kernel.invoke(
            chat_function,arguments
        )


        # Example response using context
        #print("Result_OG",result)

        return {"output" : f"{result}" }
    except Exception as e:
        return {"error": str(e)}




@app.post("/clear_data/")
async def clear_data():
    try:
        # Clear the CSV collection
        csv_docs = csv_collection.get()
        if "ids" in csv_docs and csv_docs["ids"]:
            csv_collection.delete(ids=csv_docs["ids"])
        
        # Clear the PDF collection
        pdf_docs = pdf_collection.get()
        if "ids" in pdf_docs and pdf_docs["ids"]:
            pdf_collection.delete(ids=pdf_docs["ids"])

        docs_1 = collection.get()
        if "ids" in docs_1 and docs_1["ids"]:
            collection.delete(ids=docs_1["ids"])
        
        return {"status": "success", "message": "ChromaDB collections cleared successfully!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
