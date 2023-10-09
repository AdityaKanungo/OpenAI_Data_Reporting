import streamlit as st
from PIL import Image
import openai
import json
import base64
import sqlite3
import pandas as pd
import sqlparse


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
    tables = list(metadata.keys())
    columns = {table: [col['name'] for col in metadata[table]['columns']] for table in tables}
    return {"tables": tables, "columns": columns}

def is_select_statement(query):
    parsed = sqlparse.parse(query)
    if parsed:
        return parsed[0].get_type().lower() == 'select'
    return False

def generate_sql_query(prompt, formatted_metadata):
    # Set up the conversation with the model
    messages = [
        {"role": "system", "content": f"Given a database with tables {formatted_metadata['tables']} and columns {formatted_metadata['columns']}, you need to generate SQL queries. Only give the sql query without triple `, no other text."},
        {"role": "user", "content": prompt}
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
left_column, spacing_column, right_column = st.columns([0.30, 0.02, 0.30])  # Adjust the values for desired width ratio

# Left Column: File uploader and DB details
with left_column:
    uploaded_file = left_column.file_uploader("Upload DB Metadata JSON file", type="json")
    
    if uploaded_file:
        metadata = json.load(uploaded_file)
        formatted_metadata = format_metadata_for_prompt(metadata)
        
        # Display table details
        left_column.subheader("Database Tables and Columns:")

        # Format the details
        details = ""
        for table, columns in formatted_metadata['columns'].items():
            details += f"{table}: {', '.join(columns)}\n\n"

        # Display in a code block
        left_column.code(details, language="markdown")


# ... [your previous code remains unchanged up to the point where you're generating the SQL query]

# ... [your previous code remains unchanged up to the point where you're generating the SQL query]

# Right Column: User prompt and SQL output
with right_column:
    # Get the prompt from the user
    prompt = right_column.text_area("Enter your prompt:")

    # Initialize session state variables if they do not exist
    if 'sql_query' not in st.session_state:
        st.session_state.sql_query = ""
    if 'execute_clicked' not in st.session_state:
        st.session_state.execute_clicked = False

    # If the Generate button is pressed, generate the SQL query
    if right_column.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                st.session_state.sql_query = generate_sql_query(prompt, formatted_metadata)
            right_column.code(st.session_state.sql_query, language="sql")
            
            # Check if the generated SQL is a SELECT statement
            if is_select_statement(st.session_state.sql_query):
                st.session_state.execute_clicked = False  # Reset the state of the execute button
            else:
                st.warning("Only SELECT queries can be executed here.")
                st.session_state.execute_clicked = True  # Hide the uploader for non-SELECT queries
        else:
            right_column.warning("Please enter a prompt.")

    if is_select_statement(st.session_state.sql_query) and not st.session_state.execute_clicked:
        if st.button("Execute Query"):
            st.session_state.execute_clicked = True

# If the "Execute Query" button has been clicked and it's a SELECT query, display the SQLite DB uploader
if st.session_state.execute_clicked and is_select_statement(st.session_state.sql_query):
    db_file = st.file_uploader("Upload SQLite Database file", type=["db", "sqlite", "sqlite3"])

    if db_file:
        conn = sqlite3.connect(db_file.name)  # Connect to the uploaded SQLite db
        try:
            # Execute the query and convert results to DataFrame
            df = pd.read_sql_query(st.session_state.sql_query, conn)
            st.write(df)  # Display the results
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")
        finally:
            conn.close() 