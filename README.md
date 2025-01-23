# RAGQuery: A Streamlit-Powered RAG and SQL Query Generator

## Overview
**RAGQuery** combines the power of Retrieval-Augmented Generation (RAG) with SQL query generation to offer a seamless experience for retrieving and querying data. This application enables users to: 

1. Generate SQL queries from natural language questions.
2. Perform retrieval-augmented question answering from indexed CSV and PDF documents.
3. Populate and query an SQLite database with uploaded CSV files.

This tool is ideal for developers, data analysts, and researchers who need efficient data retrieval and SQL generation capabilities.

---

## Features

### 1. SQL Query Generation
- **Natural Language Input:** Generate SQL queries by asking questions in plain English.
- **Result Display:** Execute the generated SQL queries and display results directly in the application.
- **Database Support:** Query data from an SQLite database.

### 2. RAG (Retrieval-Augmented Generation) Retrieval
- **Document Upload:** Index CSV and PDF documents for semantic search.
- **Contextual Answers:** Answer user questions by retrieving relevant data from indexed documents.
- **References:** Provide references for the retrieved context (e.g., filenames, page numbers).

### 3. CSV Database Integration
- **Database Population:** Upload CSV files to create and populate SQLite tables.
- **Table Schema:** Predefined schema for storing breast cancer data.

---

## Installation

### Prerequisites
- Python 3.8+
- [Streamlit](https://streamlit.io/) installed
- OpenAI API key (set in `.env` file)

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Usage

### SQL Query Generation
1. Navigate to the "SQL Query Generation" mode in the sidebar.
2. Enter the path to your SQLite database (default: `breast_cancer_db.sqlite`).
3. Type your question in plain English and click **Generate SQL Query**.
4. View the generated SQL query and results in the output area.

### RAG Retrieval
1. Switch to the "RAG Retrieval" mode in the sidebar.
2. Upload CSV or PDF files for indexing.
3. Type your query in the chat input and receive contextual answers.
4. View references for retrieved context, including filenames and page numbers.

### Upload CSV to Populate Database
1. Upload a CSV file under the **Upload CSV to Populate Database** section.
2. The file will be processed and the data will be stored in the SQLite database.

---

## File Structure
```
├── app.py               # Main application file
├── prep.py              # CSV preprocessing utilities
├── csvplugin.py         # Custom plugin for handling CSV data
├── requirements.txt     # Dependencies
├── data/                # Directory for storing SQLite databases
├── .env                 # Environment variables (not included in repo)
```

---

## Technologies Used
- **Streamlit**: Interactive web application framework.
- **OpenAI GPT**: Natural language processing for query generation.
- **SQLite**: Lightweight database for storing and querying data.
- **PyPDF2**: PDF text extraction for indexing.
- **Semantic Kernel**: Powering retrieval-augmented generation.

---

## Future Enhancements
1. Add support for other database types (e.g., MySQL, PostgreSQL).
2. Enable advanced filtering and customization of RAG search results.
3. Introduce additional file types (e.g., Excel, Word) for indexing.
4. Improve error handling and user experience.

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
Special thanks to the open-source libraries and frameworks that make this project possible.
