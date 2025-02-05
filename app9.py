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
from semantic_kernel.functions import KernelArguments
import pandas as pd
import re

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

def get_context(user_input, file_type):
    try:
        # Pass the file type as a prefix to the query
        payload = {"query": f"{file_type}:{user_input}"}
        print("Payload:", payload)
        response = requests.post(get_url("context"), json=payload)
        print("Response:", response)
        response.raise_for_status()
        data = response.json()
        docs = [str(doc) for doc in data["docs"]]  # Convert each dictionary to a string
        return "\n".join(docs), data["filenames"], data["page_number"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching context: {e}")
        return None, None, None

def get_response(user_input, file_type):
    context, filenames, page_numbers = get_context(user_input, file_type)
    if not context:
        return "Unable to fetch context.", filenames, page_numbers, chat_history

    try:
        payload = {
            "query": user_input,
            "context": context
        }
        response = requests.post(get_url("autogen_response"), json=payload)
        response.raise_for_status()

        # Extract response JSON
        result = response.json()
        autogen_output = result.get("output", "No response received.")
        chat_history.extend(result.get("chat_history", []))  # Append chat history

        return autogen_output, filenames

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching response: {e}")
        return "Unable to fetch response.", filenames





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

    # Test retrieval 
    data = pd.read_sql("SELECT * FROM breast_cancer_data LIMIT 5", conn)
    print("Data inserted successfully!")
    #print(data)

    # Close connection
    conn.close()


# Initialize Database
st.set_page_config(page_title="Agentic AI for Insurance", layout="wide")

# Sidebar
with st.sidebar:

    
    st.sidebar.title("⚙️ Options")
    option = st.radio("Choose a Mode", ["🔍 SQL Query Generation", "🧠 RAG Retrieval"])
    if st.sidebar.button("🧹 Clear ChromaDB"):
        with st.spinner("Clearing ChromaDB..."):
            try:
                response = requests.post(get_url("clear_data"))
                response.raise_for_status()
                result = response.json()
                if result["status"] == "success":
                    st.success(result["message"])
                else:
                    st.error(result["message"])
            except requests.exceptions.RequestException as e:
                st.error(f"Clearing ChromaDB: {e}")
st.title("🔍 AI-Powered Insurance Processing")
# Main Application
if option == "🔍 SQL Query Generation":
    st.title("SQL Query Generator 💬")
    db_path = st.text_input("SQLite DB Path", BREAST_CANCER_DB)
    question = st.text_input("Enter your question:")

    if st.button("Generate SQL Query"):
        if not os.path.exists(db_path):
            st.error("Database file does not exist.")
        else:
            with st.spinner("Generating SQL Query and fetching results..."):
                payload = {"question": question, "db_path": db_path}
                response = requests.post(get_url("sql_query"), json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    print(result)
                    st.text_area("Generated SQL Query & Results", result.get("result", "Error fetching results"),height = 100)
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error')}")


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

elif option == "🧠 RAG Retrieval":
    st.title("RAG Question Answering 💬")
    uploaded_files = st.file_uploader(
        "Choose CSV or PDF files", type=["csv", "pdf"], accept_multiple_files=True
    )
    up_name = [file.name for file in uploaded_files] if uploaded_files else []
    if uploaded_files:
        if "files" not in st.session_state:
            st.session_state["files"] = []
        if up_name[0].endswith(".csv"):
            file_type = "csv"
        else:
            file_type = "pdf"

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

    chat_history = []
    #user_input = st.chat_input("Your Message:")
    user_input = st.chat_input("Your Message:")
    chat_history.append(user_input)
    if user_input:
        st.chat_message("user").write(user_input)

        # Fetch AutoGen response
        response, filenames= get_response(user_input, file_type)

        # Format response for readability
        st.markdown("### 🤖 AutoGen Response")
        #st.markdown(response.replace("\n", "\n\n"))  # Add spacing
        #bot_response = f"{response}\n"
        #print(bot_response)
        
        # Assuming bot_response contains the JSON output as a string
        #bot_response = f"{response}\n".strip()

        
        # Assuming bot_response contains JSON output as a string
        bot_response = f"{response}\n".strip()
        print(type(bot_response))
        bot_response_fixed = bot_response.replace("'", '"').replace("None", "null")


        print(repr(bot_response_fixed))  # Shows special characters and formatting issues
    
        response_dict = {}
        response_dict = json.dumps(bot_response_fixed)

        #bot_response = bot_response.replace("'", '"')  # Add spacing


    # Print dictionary format


    

        st.session_state.chat_history.append((bot_response, True))
        st.chat_message("assistant").write(bot_response)

        # Display references if available
        if filenames:
            st.markdown("#### 📂 References:")
            for file in filenames:
                st.markdown(f"- {file}")

        # Add a divider for clarity
        st.divider()
