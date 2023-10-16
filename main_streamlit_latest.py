import streamlit as st
from PIL import Image
import openai
import json
import base64
import sqlite3
import pandas as pd
import sqlparse
from sqlalchemy import create_engine

# Streamlit UI Configuration
st.set_page_config(page_title="SQL Query Generator", layout="wide")

## Header - Nav --------------------------

def get_image_as_data_url(file_path):
    with open(file_path, "rb") as image_file:
        # Convert binary image to base64 string
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Return image as a data URL
    return f"data:image/png;base64,{encoded_image}"

def get_custom_html(data_url):
    with open("custom_styles.html", "r") as file:
        return file.read().replace("{data_url}", data_url)

# Get the image as a data URL
data_url = get_image_as_data_url("C:/Users/aksha/Desktop/Test db/header3.png")

# Inject custom CSS and HTML from file
custom_html = get_custom_html(data_url)
st.markdown(custom_html, unsafe_allow_html=True)


# Create a container with the class "navbar"
st.markdown(
    """
    <div class='navbar'>
        <p></p>
    </div>
    """,
    unsafe_allow_html=True,
)

### End Header - nav -----------------------

# Initialize the OpenAI API (for security reasons, it's better to set this as an environment variable)
openai.api_key = ''

def format_metadata_for_prompt(metadata):
    if isinstance(metadata, list):
        tables = [item['name'] for item in metadata]
        columns = {table['name']: [col['name'] for col in table['columns']] for table in metadata}
    else:
        tables = list(metadata.keys())
        columns = {table: [col['name'] for col in metadata[table]['columns']] for table in tables}
    return {"tables": tables, "columns": columns}


# Add a function to connect to the database
def connect_to_db(host, port, user, password, dbname):
    # This is for SQLite; for other databases, modify the connection string
    conn_str = f"sqlite:///{dbname}"
    return sqlite3.connect(conn_str)


def is_select_statement(query):
    parsed = sqlparse.parse(query)
    if parsed:
        return parsed[0].get_type().lower() == 'select'
    return False

def generate_sql_query(prompt, formatted_metadata):
    # Set up the conversation with the model
    messages = [

{"role": "system", "content": f"""Based on the provided metadata, generate a SQL query in accordance with the following rules:

Metadata

Tables and columns:

Tables: {formatted_metadata['tables']}

Columns: {formatted_metadata['columns']}

User's Prompt: {prompt}

custom styles.html

output.json

Please adhere to these guidelines:

IMPORTANT: If a request involves attributes, columns or relationships NOT present in the provided metadata, DO NOT genaret SQL query for it. Instead, respond with "The requested data is not available in the provided metadata"

1. The Tables and Columns provided are the only permissible entities to be used in the SQL. Do not assume or introduce any table or column not mentionend explicitly in the meadata.

2. Do not expand upon or make assumptions about the database schema based on the user's prompt. If the user asks for the data not in the provided metadata, it should be considered out of scope.

3. Ensure every column referred to in the SQL is directly tracebale back to the provided metadata (Tables/Columns). No columns should be synthesized or assumed based on context. 4. Do not create or reference temporary, derieved, or virtual tables that aren't explicitly part of the provided metadata.

5. If a user prompt can be interpreted in multiple ways, cross-reference with the metadata to choose the interpretation that matches the available tables and columns. 6. Do not make assumptions about the relationship between tables unless it is clear from the metadata

7. If the user's prompt is at odds with the provided metadata, prioritize the metadatas constraints. return a note or an error about the discrepancy rather than trying to fill the gaps with assumptions. 8. Avoid generating subqueries that intoduce tables or columns not present in the provided metadata, even if they might seem contextually relevant to the user's prompt.

9. If a user prompt request data or relationship not covered in the metadata, instead of making assumptions or introductions, consider providing a transparent message or feedack indicating the mismatch.

10. The SQL should be up to general standards and easy understandable

11. Use consistent SQL syntax and formatting, maintaning the appropriate use of whitespace and indentation for redablity

12. Do not include any feature, table or column, that is not provided in the given list 13. Ensure the query is accurate and does not have errors

14. Use aliases for tables and sub-queries to enhance readability and ensure clarity in referencing. 15. Avoid using wildcards (") for selecting columns. Specify the exact column names required.

16. Ensure that the queries are optimized for performance, ensuring that any joins or where clauses are using indexed columns wherever possible. 17. Avoid using nested queries unless absolutely necessary. and if used, they should not be overly complex or deeply nested.

18. Use explicit 30IN' clauses instead of implict onces to improve readablity and ensure clarity in relationships between tables. 19. Do not use aggrehation funcations like 'SUM', 'AVG', etc unless the user's prompt explicitly asks for an aggregated result.

20. Ensure that the selected columns are relevant to the user's request and donot retrieve unnecessary data. 21. Where possible, encourage filtering data using 'WHERE' clause to limit data retrival to only what's necessary per user's request.

22. The query should have a single responsiblity and not try to accomplish multiple unrelated tasks


"""}
    ]
    
    # Use OpenAI to generate SQL using the chat endpoint
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages
    )
    
    # Extract the model's message from the response
    return response.choices[0].message['content'].strip()

# Streamlit UI Configuration
#st.set_page_config(page_title="SQL Query Generator", layout="wide")

# Add logo to the left column (You can adjust the width as needed)
image = Image.open('Capture.png')
st.image(image, width=200)

st.title("SQL Query Generator using OpenAI")

# Create columns with custom widths and a spacing column in between
left_column, spacing_column, right_column = st.columns([0.40, 0.015, 0.20])  # Adjust the values for desired width ratio

# Left Column: File uploader and DB details
with left_column:
    uploaded_file = left_column.file_uploader("Upload DB Metadata JSON file", type="json")
    
    if uploaded_file:
        metadata = json.load(uploaded_file)
        formatted_metadata = format_metadata_for_prompt(metadata)
        
        # Display table details
        left_column.subheader("Database Tables and Columns:")

        # Format the details
        details = []
        for table, columns in formatted_metadata['columns'].items():
            # Color the table names with a shade of blue and columns with a shade of green
            line = f"<div style='margin-top: 0; padding: 0;'><strong style='color: #007ACC;'>{table}</strong>: <span style='color: #FFFFFF;'>{', '.join(columns)}</span></div>"
            details.append(line)

        # Join the details and display in the markdown with controlled spacing, background color, and left padding
        left_column.markdown(f"""
        <div style='white-space: pre-wrap; overflow-wrap: break-word; font-family: "Courier New", monospace; background-color: #253547; padding: 10px; border-radius: 5px; margin: 0; padding: 0;'>
        <div style='margin-top: 0; padding-left: 1em;'>{''.join(details)}</div>
        </div>
        """, unsafe_allow_html=True)


with right_column:
    # Get the prompt from the user
    prompt = right_column.text_area("Enter your prompt:")

    # Initialize session state variables if they do not exist
    if 'sql_query' not in st.session_state:
        st.session_state.sql_query = ""
    if 'show_execute_button' not in st.session_state:
        st.session_state.show_execute_button = False

    # Generate SQL button
    if right_column.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                st.session_state.sql_query = generate_sql_query(prompt, formatted_metadata)
            st.write('')
            #right_column.code(st.session_state.sql_query, language="sql")
            
            if is_select_statement(st.session_state.sql_query):
                st.session_state.show_execute_button = False


    # ... [Rest of the code]
    # Display DB Connection details and Connect Database button after SQL is generated
    if st.session_state.sql_query:
        right_column.code(st.session_state.sql_query, language="sql")

        if is_select_statement(st.session_state.sql_query):
            st.write("DB Connection Details:")

            # First row
            col1, col2 = st.columns(2)
            db_host = col1.text_input("Host", key="db_host")
            db_port = col2.text_input("Port", key="db_port")

            # Second row
            col1, col2 = st.columns(2)
            db_user = col1.text_input("Username", key="db_user")
            db_pass = col2.text_input("Password", type="password", key="db_pass")

            # Third row
            col1, col2 = st.columns(2)
            db_name = col1.text_input("Database Name", key="db_name")

            if st.button("Connect Database"):
                try:
                    conn_str = f"mssql+pyodbc://{db_user}:{db_pass}@{db_host}\\SQLEXPRESS:{db_port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server"
                    engine = create_engine(conn_str)
                    st.session_state.connection = engine.connect()  # Store the connection in session_state
                    st.success("Connected successfully!")
                    st.session_state.show_execute_button = True
                except Exception as e:
                    st.error(f"Error establishing connection: {str(e)}")
        else:
            st.warning("Only SELECT statements can be executed.")

    # Execute Query button
    if st.session_state.show_execute_button:
        if st.button("Execute Query"):
            try:
                df = pd.read_sql_query(st.session_state.sql_query, st.session_state.connection)  # Use the connection from session_state
                st.session_state.query_executed = True  # Set the flag
                st.session_state.query_result = df  # Store the result in session_state
            except Exception as e:
                st.error(f"Error executing query: {str(e)}")
                st.session_state.query_executed = False  # Ensure flag is set to False in case of error

# Now, outside the column contexts, check for the flag and display the results.
if 'query_executed' in st.session_state and st.session_state.query_executed:
    st.write('')
    st.write('')
    st.subheader('Results')
    st.write(st.session_state.query_result)


