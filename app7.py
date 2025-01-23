import os
import textwrap
import streamlit as st
from dotenv import load_dotenv
import asyncio
import requests
from tqdm import tqdm
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAITextEmbedding
from prep import load_and_process_csv
from csvplugin import CSVPlugin
import json
import PyPDF2
import sqlite3
import pandas as pd

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Please check your environment variables.")

# SQLite DB Path
DB_PATH = "data.db"
BREAST_CANCER_DB = "breast_cancer_db.sqlite"

# Kernel setup for RAG
kernel = Kernel()
kernel.add_plugin(CSVPlugin(), plugin_name="CSV")

# Helper Functions for RAG
def get_url(endpoint):
    return f"http://127.0.0.1:8000/{endpoint}/"

def get_context(user_input):
    try:
        payload = {"query": user_input}
        print("Payload:", payload)
        response = requests.post(get_url("context"), json=payload)
        print("Response:", response)
        response.raise_for_status()
        data = response.json()
        print("Data:", data)
        #print("Docs:",data)
        docs = [str(doc) for doc in data["docs"]]  # Convert each dictionary to a string
        return "\n".join(docs), data["filenames"], data["page_number"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching context: {e}")
        return None, None, None

def get_response(user_input):
    context, filenames, page_numbers = get_context(user_input)
    print("Context:", context)
    if not context:
        return "Unable to fetch context.", filenames, page_numbers
    try:
        payload = {"query": user_input, "context": context}
        response = requests.post(get_url("response"), json=payload)
        response.raise_for_status()
        return response.json()["output"], filenames, page_numbers
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching response: {e}")
        return "Unable to fetch response.", filenames, page_numbers
def create_and_populate_table(uploaded_file):
    """Create and populate the breast cancer data table with the uploaded file."""
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Connect to SQLite database
    conn = sqlite3.connect("breast_cancer_db.sqlite")
    cursor = conn.cursor()

    # Create the table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS breast_cancer_data (
        id INTEGER PRIMARY KEY,
        diagnosis TEXT,
        radius_mean REAL,
        texture_mean REAL,
        perimeter_mean REAL,
        area_mean REAL,
        smoothness_mean REAL,
        compactness_mean REAL,
        concavity_mean REAL,
        concave_points_mean REAL,
        symmetry_mean REAL,
        fractal_dimension_mean REAL,
        radius_se REAL,
        texture_se REAL,
        perimeter_se REAL,
        area_se REAL,
        smoothness_se REAL,
        compactness_se REAL,
        concavity_se REAL,
        concave_points_se REAL,
        symmetry_se REAL,
        fractal_dimension_se REAL,
        radius_worst REAL,
        texture_worst REAL,
        perimeter_worst REAL,
        area_worst REAL,
        smoothness_worst REAL,
        compactness_worst REAL,
        concavity_worst REAL,
        concave_points_worst REAL,
        symmetry_worst REAL,
        fractal_dimension_worst REAL
    );
    """)
    conn.commit()

    # Insert the data into the table
    df.to_sql("breast_cancer_data", conn, if_exists="replace", index=False)

    # Test retrieval (optional for debugging)
    data = pd.read_sql("SELECT * FROM breast_cancer_data LIMIT 5", conn)
    print("Data inserted successfully!")
    #print(data)

    # Close connection
    conn.close()



def generate_sql_query_and_fetch_result(question: str, db_path: str) -> str:
    prompt = f"""
    You are an expert in SQL. Generate a valid SQL query for the following question. Do not include explanations, comments, or additional text.
    Question: {question}
    """
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        },
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0,
        },
    ).json()

    sql_query = response["choices"][0]["message"]["content"].strip()

    # Execute the SQL query
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        conn.close()

        # Format results
        result_table = f"SQL Query:\n{sql_query}\n"
    except Exception as e:
        conn.close()
        return f"SQL Query:\n{sql_query}\n\nError:\n{str(e)}"
    
    return result_table

# Initialize Database


# Sidebar
with st.sidebar:

    
    st.title("Options")
    option = st.radio("Choose a Mode", ["SQL Query Generation", "RAG Retrieval"])

# Main Application
if option == "SQL Query Generation":
    st.title("SQL Query Generator 💬")
    db_path = st.text_input("SQLite DB Path", BREAST_CANCER_DB)
    question = st.text_input("Enter your question:")
    if st.button("Generate SQL Query"):
        if not os.path.exists(db_path):
            st.error("Database file does not exist.")
        else:
            with st.spinner("Generating SQL Query and fetching results..."):
                result = generate_sql_query_and_fetch_result(question, db_path)
                st.text_area("Generated SQL Query & Results", result, height=300)


    # Replace the CSV upload logic with this new function
    st.subheader("Upload CSV to Populate Database")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        try:
            with st.spinner("Uploading and processing the CSV file..."):
                create_and_populate_table(uploaded_file)
            st.success(f"Data from {uploaded_file.name} has been successfully uploaded to the database!")
        except Exception as e:
            st.error(f"Failed to process the uploaded file: {e}")

elif option == "RAG Retrieval":
    st.title("RAG Question Answering 💬")
    uploaded_files = st.file_uploader(
        "Choose CSV or PDF files", type=["csv", "pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        if "files" not in st.session_state:
            st.session_state["files"] = []

        with st.spinner("Indexing documents..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state["files"]:
                    try:
                        # Process files
                        if uploaded_file.name.endswith(".csv"):
                            print("Done")
                            csv_data = asyncio.run(load_and_process_csv(uploaded_file))
                            print("CSV Data:", csv_data)
                            file_type = "csv"
                        elif uploaded_file.name.endswith(".pdf"):
                            pdf_data = []
                            reader = PyPDF2.PdfReader(uploaded_file)
                            for page_number, page in enumerate(reader.pages, start=1):
                                text = page.extract_text()
                                pdf_data.append(
                                    {
                                        "index": page_number,
                                        "data": {"text": text},
                                        "filename": uploaded_file.name,
                                        "page_number": page_number,
                                    }
                                )
                            csv_data = pdf_data
                            file_type = "pdf"

                        # Index documents
                        for i in tqdm(range(0, len(csv_data), 10)):
                            payload = {
                                "items": [
                                    {
                                        "id": str(item["index"]),
                                        "line": json.dumps(item["data"]),
                                        "filename": uploaded_file.name,
                                    }
                                    for item in csv_data
                                ]
                            }
                            #print("payload:", payload)
                            response = requests.post(get_url("push_docs"), json=payload)
                            print("Response:", response)
                            response.raise_for_status()

                        st.session_state["files"].append(uploaded_file.name)
                    except Exception as e:
                        st.error(f"Failed to index {uploaded_file.name}: {e}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Your Message:")
if user_input:
    # Display user messages
    for message, is_bot_response in st.session_state.chat_history:
        if is_bot_response:
            st.chat_message("assistant").write(message)
        else:
            st.chat_message("user").write(message)

    st.session_state.chat_history.append((user_input, False))
    st.chat_message("user").write(user_input)

    # Fetch response
    response, filenames, page_numbers = get_response(user_input)

    # Construct bot response with references
    bot_response = f"{response}\n\nReferences:\n{uploaded_file.name}\n"
   

    st.session_state.chat_history.append((bot_response, True))
    st.chat_message("assistant").write(bot_response)

