import streamlit as st
from PIL import Image
import openai
import json
import base64
import io
import sqlite3
import ast
import pandas as pd
import re
import squarify
import seaborn as sns
import matplotlib.pyplot as plt
import sqlparse
from sqlalchemy import create_engine

# Streamlit UI Configuration
st.set_page_config(page_title="SQL Query Generator", layout="wide")

## Header - Nav --------------------------

def get_image_as_data_url(file_path):
    with open(file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded_image}"

def get_custom_html(data_url):
    with open("custom_styles.html", "r") as file:
        return file.read().replace("{data_url}", data_url)

data_url = get_image_as_data_url("header3.png")
custom_html = get_custom_html(data_url)
st.markdown(custom_html, unsafe_allow_html=True)
st.markdown(
    """
    <div class='navbar'>
        <p></p>
    </div>
    """,
    unsafe_allow_html=True,
)

### End Header - nav -----------------------

#-----------------------------------------------


import openai

def generate_description_with_openai(metadata):
    try:
        # Count the number of tables
        num_tables = len(metadata)
        
        # Create lists to store table names, column names, and column types
        table_names = []
        all_columns = []
        column_types = []
        
        # Iterate over the tables in the metadata
        for table_name, table_info in metadata.items():
            table_names.append(table_name)
            if 'columns' in table_info:
                for column in table_info['columns']:
                    all_columns.append(column['name'])
                    if 'type' in column:
                        column_types.append(column['type'])
        
        # Check if there are enough tables to generate a description
        if num_tables == 0:
            return "The metadata does not contain enough information about the tables to generate a description."

        # Create a prompt for the OpenAI API
        prompt = (
            f"""As a seasoned data analyst examining a database consisting of {num_tables} tables, namely {', '.join(table_names)}, 
            I have observed a variety of attributes across these tables. 
            The database encompasses a wide range of columns, including {', '.join(table_names)}. 
            Could you assist me in articulating a comprehensive and insightful summary, 
            shedding light on the potential contents, overarching themes, and the unique aspects of this database? 
            I am aiming for a narrative that offers a holistic and nuanced understanding of how this database could be leveraged for analytical purposes.
            Give a strictly 30 words description only.
            """
        )

        # Call the OpenAI API
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        
        # Extract the response text
        description = response['choices'][0]['text'].strip()
        return description
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "An error occurred while generating the description."





#------------------------------------------------



# Initialize the OpenAI API
openai.api_key = ""


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


def get_table_download_link(df, filename="results.xlsx", text="Download results"):
    """Generates a link allowing the data in a given panda dataframe to be downloaded"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1')
    binary_data = output.getvalue()
    b64 = base64.b64encode(binary_data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'



def is_select_statement(query):
    parsed = sqlparse.parse(query)
    if parsed:
        return parsed[0].get_type().lower() == 'select'
    return False


import spacy

nlp = spacy.load("en_core_web_sm")

def is_prompt_relevant_to_metadata(prompt, formatted_metadata):
    # Define your own stop words
    custom_stop_words = {"is", "are", "the", "of", "and", "a", "to", "in", "that", "it", "with", "as", "for", "on", "was", "at", "by", "an", "be", "this", "which", "or", "from", "but", "not", "can", "they", "their", "you", "have", "had", "all", "my", "we", "do", "if", "me", "so", "what", "him", "your", "no", "there", "when", "up", "out", "who", "them", "then", "she", "each", "would", "about", "how", "other", "into", "could", "her", "has", "said", "just", "some", "like", "any", "more", "also", "now", "see", "only", "his", "been", "because", "did", "get", "come", "made", "two", "over", "know", "use", "way", "even", "new", "after", "us", "time", "many", "well", "may", "where", "most", "these", "very", "before", "need", "here", "much", "those", "must", "such", "why", "going", "big", "through", "our", "part", "little", "never", "around", "went", "people", "take", "long", "still", "find", "own", "down", "keep", "every", "another", "being", "once", "does", "got", "really", "say", "might", "always", "things", "want", "give", "using", "goes", "think", "better", "great", "right", "thing", "yes", "back", "should", "off", "take", "while", "same", "around", "going", "thing", "give", "using", "think", "better"}


    # Tokenize the prompt
    prompt_doc = nlp(prompt.lower())
    
    # Include tokens that are alpha characters and not in the custom stop words list,
    # or tokens that are part of the metadata names
    all_names = set()
    for table, columns in formatted_metadata['columns'].items():
        all_names.add(table.lower())
        for column in columns:
            all_names.add(column.lower())
    
    prompt_tokens = [nlp(token.text)[0] for token in prompt_doc if token.is_alpha and (token.text not in custom_stop_words or token.text in all_names)]
    
    # Calculate the relevance score based on similarity
    relevance_score = 0
    for token in prompt_tokens:
        similarities = [token.similarity(nlp(name)[0]) for name in all_names]
        relevance_score += max(similarities) if similarities else 0

    relevance_score /= len(prompt_tokens) if prompt_tokens else 0
    relevance_score = round(relevance_score,2)

    relevance_threshold = 0.85
    st.write('Relevance Score:', relevance_score)
    #st.write('Prompt Tokens:', set(prompt_tokens))
    #st.write('Metadata Names:', all_names)
    
    if relevance_score >= relevance_threshold:
        return 1
    else:
        return 2


def generate_sql_query(prompt, formatted_metadata):
    # Check if the prompt is relevant to the metadata
    messages = [

{"role": "system", "content": f"""

Generate a SQL SELECT query that is compatible with MS SQL and achieves the OBJECTIVE exclusively using only the tables and views described in "SCHEMA:".

Only generate SQL if the OBJECTIVE can be answered by querying a database with tables described in SCHEMA.

Do not include any explanations, only provide valid SQL.

[BEGIN EXAMPLE]

SCHEMA:
description: historical record of concerts, stadiums and singers
tables:
  - stadium: 
    columns:
      Stadium_ID: 
      Location: 
      Name: 
      Capacity: 
      Highest: 
      Lowest: 
      Average: 
  - singer: 
    columns:
      Singer_ID: 
      Name: 
      Country: 
      Song_Name: 
      Song_release_year: 
      Age: 
      Is_male: 
  - concert: 
    columns:
      concert_ID: 
      concert_Name: 
      Theme: 
      Stadium_ID: 
      Year: 
  - singer_in_concert: 
    columns:
      concert_ID: 
      Singer_ID: 
references:
  concert.Stadium_ID: stadium.Stadium_ID
  singer_in_concert.concert_ID: concert.concert_ID
  singer_in_concert.Singer_ID: singer.Singer_ID

OBJECTIVE: "How many heads of the departments are older than 56 ?"
SQL: select count(*) department_head_count from head where age > 56

[END EXAMPLE]

SCHEMA:
Tables and columns:
Tables: {formatted_metadata['tables']}
Columns: {formatted_metadata['columns']}

OBJECTIVE: {prompt}
SQL: Let's think step by step.
The final output should have just the sql statement without any comments or additional description

"""}
    ]
    
    # Use OpenAI to generate SQL using the chat endpoint
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages
    )
    
    # Extract the model's message from the response
    return response.choices[0].message['content'].strip()



def generate_query_description(generated_sql, formatted_metadata):
    # Formatting the metadata for better readability and understanding
    tables_str = ", ".join(formatted_metadata['tables'])
    columns_dict = formatted_metadata['columns']
    columns_str = ", ".join([f"{table}: {', '.join(columns)}" for table, columns in columns_dict.items()])
    
    # Set up the conversation with the model
    messages = [
        {"role": "system", "content": f"""
Please generate stricktly 30 words description for the following SQL query based on the provided metadata.

SQL Query:
{generated_sql}

Metadata:
Tables: {tables_str}
Columns: {columns_str}
"""}
    ]
    
    # Use OpenAI to generate the description
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages
    )
    
    # Extract the model's message from the response
    description = response.choices[0]['message']['content'].strip()
    
    return description

#-----------------------------------------------------------------------
def query_result_data_with_openai(prompt, df_columns):
    # Set up the conversation with the model
    columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in df_columns.items()])
    messages = [
        {"role": "system", "content": "You are provided with a dataframe 'df' that has the following columns and their data types: \n" + columns_info + "\nAnswer questions based on this and provide Python code for plotting using any of these if required : numpy, pandas, plotly, and matplotlib. Add colors to the plots. Use st.plotly_chart(fig) if using plotly"},
        {"role": "user", "content": prompt}
    ]

    # Use OpenAI to generate a response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Extract the model's message from the response
    full_response = response.choices[0].message['content'].strip()
    
    # Use regex to find code blocks enclosed in triple backticks
    code_blocks = re.findall(r"```python(.*?)```", full_response, re.DOTALL)
    
    if code_blocks:
        python_code = code_blocks[0].strip()
        # Format the Python code
        python_code = '\n'.join(line.strip() for line in python_code.split('\n') if line.strip())
        answer = re.sub(r"```python.*?```", "", full_response, flags=re.DOTALL).strip()
    else:
        answer = full_response
        python_code = None

    return answer, python_code


#---------------------------------------------------------------------------------
#   UI
#---------------------------------------------------------------------------------

# UI Components
image = Image.open('Capture.png')
st.image(image, width=200)
st.title("SQL and Report Generation using Open AI")

# Adjust the column widths
left_column, spacing_column1 , middle_column, spacing_column2 ,right_column = st.columns([0.16, 0.010, 0.20, 0.010, 0.20])
#middle_column ,right_column = st.columns([0.080, 0.20])

with left_column:
    uploaded_file = left_column.file_uploader("Upload DB Metadata JSON file", type="json")
    if uploaded_file:
        # Check if the file has changed since the last upload
        if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            metadata = json.load(uploaded_file)
            st.session_state.formatted_metadata = format_metadata_for_prompt(metadata)
            
            # Call the API to get the data description
            data_summary = generate_description_with_openai(metadata)
            
            # Store the description and file name in the session state
            st.session_state.data_summary = data_summary
            st.session_state.uploaded_file_name = uploaded_file.name
        else:
            # If the file has not changed, use the stored description
            data_summary = st.session_state.data_summary
    else:
        # If no file is uploaded, clear the stored description and file name
        if 'data_summary' in st.session_state:
            del st.session_state.data_summary
        if 'uploaded_file_name' in st.session_state:
            del st.session_state.uploaded_file_name
        if 'formatted_metadata' in st.session_state:
            del st.session_state.formatted_metadata
        data_summary = None

    if data_summary:
        st.info(data_summary)


st.markdown("""
    <style>
        .stAlert > div {
            font-size: 4px !important;
        }
    </style>
""", unsafe_allow_html=True)

with middle_column:
    prompt = middle_column.text_area("Enter the question for the data you are looking for:")
    if 'sql_query' not in st.session_state:
        st.session_state.sql_query = ""
    if 'show_execute_button' not in st.session_state:
        st.session_state.show_execute_button = False
    if middle_column.button("Generate SQL"):
        if prompt and 'formatted_metadata' in st.session_state:
            with st.spinner("Checking prompt relevance..."):
                relevance_code = is_prompt_relevant_to_metadata(prompt, st.session_state.formatted_metadata)
                
            if relevance_code == 2:
                st.error("The prompt does not seem to be relevant to the metadata.")
            else:
                with st.spinner("Generating response..."):
                    st.session_state.sql_query = generate_sql_query(prompt, st.session_state.formatted_metadata)
                    if st.session_state.sql_query.startswith("ERROR"):
                        st.error(st.session_state.sql_query)
                    else:
                        if 'data_summary' in st.session_state:
                            st.session_state.sql_description = generate_query_description(st.session_state.sql_query, st.session_state.formatted_metadata)


    # Add an image below
    #image_path = "Capture21342.png"  # Change this to the path of your image
    #st.image(image_path, use_column_width='always')

with right_column:
    # Display the generated SQL Query
    if st.session_state.sql_query:
        right_column.write('<p style="font-size: 0.9em; margin-bottom: 5px;">Generated SQL Query:</p>', unsafe_allow_html=True)
        right_column.code(st.session_state.sql_query, language="sql")

        if 'sql_description' in st.session_state and st.session_state.sql_description:
            #right_column.write('<p style="font-size: 0.9em; margin-top: 20px; margin-bottom: 5px;">SQL Query Description:</p>', unsafe_allow_html=True)
            right_column.info(st.session_state.sql_description)
        
        # Check if the query is a SELECT statement
        if is_select_statement(st.session_state.sql_query):
            # If it is, handle database connection and set the flag to not show the warning message
            # Existing database connection code goes here...
            db_host = "localhost"
            db_port = "50003"
            db_user = "test1"
            db_pass = "test1"
            db_name = "test1"
            try:
                conn_str = f"mssql+pyodbc://{db_user}:{db_pass}@{db_host}\\SQLEXPRESS:{db_port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server"
                engine = create_engine(conn_str)
                st.session_state.connection = engine.connect()
                st.session_state.show_execute_button = True
                st.session_state.show_non_select_warning = False
            except Exception as e:
                st.error(f"Error establishing connection: {str(e)}")
                st.session_state.show_execute_button = False
        else:
            # If it is not a SELECT statement, set the flag to show the warning message when "Execute Query" is pressed
            st.session_state.show_non_select_warning = True
    else:
        # If no SQL query is generated yet, display a placeholder
        right_column.write('<p style="font-size: 0.9em; margin-bottom: 5px;">Generated SQL Query:</p>', unsafe_allow_html=True)
        right_column.code("""""", language="sql")



    execute_button = right_column.button("Execute Query")
    
    if execute_button:
        # Check if the query is a SELECT statement
        if is_select_statement(st.session_state.sql_query):
            try:
                # If it is, execute the query
                df = pd.read_sql_query(st.session_state.sql_query, st.session_state.connection)
                df.index = df.index + 1
                st.session_state.query_executed = True
                st.session_state.query_result = df
                st.session_state.query_data_clicked = False
            except Exception as e:
                st.error("There was an error executing the query, try rephrasing your question to generate new SQL.")
                
                # Extract main error line
                match = re.search(r"\[SQL Server\](.*?)\(\d+\)", str(e))
                if match:
                    main_error_line = match.group(1).strip()
                    st.error(main_error_line)
                else:
                    st.error("An unexpected error occurred. Please check the query and try again.")
                
                st.session_state.query_executed = False
        else:
            # If it is not a SELECT statement, show a warning message
            st.warning("I can only execute SELECT statements")


# Results and Data Tables Tabs
with st.container():
    tab1, tab2 = st.tabs(["Results", "Metadata"])
    
    if 'query_executed' in st.session_state and st.session_state.query_executed:
        with tab1:
            col1, col2 = st.columns([1, 10])
            result_length = len(st.session_state.query_result)
            options = [10, 50, 100, 200, 500, 1000]
            options = [option for option in options if option <= result_length] + [result_length]
            with col1:
                num_rows = st.selectbox('Select rows:', options=options, index=0, format_func=lambda x: f"All {x}" if x == result_length else str(x))

            st.write('<p style="font-size: 1em; margin-bottom: 0px;"><b>Query Results:</b></p>', unsafe_allow_html=True)

            # Define the CSS styles
            style = """
            <style>
                .dataframe-container {
                    width: 100%;  /* Set width to 100% */
                    background-color: #4e7496;  /* Background color */
                    color: white;  /* Text color */
                    border-collapse: collapse;
                    margin-bottom: 1em;
                }
                .dataframe-container th, .dataframe-container td {
                    border: 1px solid black;
                    padding: 8px;
                    text-align: left;
                }
                .dataframe-container th {
                    background-color: #253547;  /* Header background color */
                }
                .scrollable-container {
                    overflow-x: auto;  /* Make horizontally scrollable */
                    overflow-y: auto;  /* Make vertically scrollable */
                    max-height: 500px;  /* Adjust as needed */
                }
            </style>
            """

            with st.container():
                # Convert dataframe to HTML and then use st.markdown to display it
                df_html = st.session_state.query_result.head(num_rows).to_html(classes='dataframe-container', escape=False)
                # Display the dataframe with styles
                st.markdown(style, unsafe_allow_html=True)
                st.markdown("<div class='scrollable-container'>" + df_html + "</div>", unsafe_allow_html=True)

            st.markdown(get_table_download_link(st.session_state.query_result), unsafe_allow_html=True)
            st.subheader("Analyze Result Data")


            with tab2:
                        if 'formatted_metadata' in st.session_state:
                            st.subheader("Database Tables and Columns:")
                            details = []
                            for table, columns in st.session_state.formatted_metadata['columns'].items():
                                line = f"<div style='margin-top: 0; padding: 0;'><strong style='color: #007ACC;'>{table}</strong>: <span style='color: #FFFFFF;'>{', '.join(columns)}</span></div>"
                                details.append(line)
                            st.markdown(f"""
                            <div style='white-space: pre-wrap; overflow-wrap: break-word; font-family: "Courier New", monospace; background-color: #253547; padding: 10px; border-radius: 5px; margin: 0; padding: 0;'>
                            <div style='margin-top: 0; padding-left: 1em;'>{''.join(details)}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning("Please upload the database metadata JSON file to view the database tables and columns.")

    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    def get_embeddings(text_list):
        response = openai.Embedding.create(input=text_list, model="text-embedding-ada-002")
        # Extract the numerical embeddings from the response data
        return [embedding['embedding'] for embedding in response['data']]

    def find_most_relevant_data(embeddings, query_embedding):
        # Calculate cosine similarities between the query and each data embedding
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        # Find the index of the highest similarity score
        most_relevant_idx = np.argmax(similarities)
        
        # Return the most relevant data point based on the highest similarity score
        return most_relevant_idx

    # Initialize session state
    if 'query_data_clicked' not in st.session_state:
        st.session_state.query_data_clicked = False

    # User input for query
    result_query_prompt = st.text_area("Enter your question for additional analysis:")

    # Button to trigger query
    if st.button("Generate Results"):
        st.session_state.query_data_clicked = not st.session_state.query_data_clicked

    # Handle query and display results
    if st.session_state.query_data_clicked and result_query_prompt:
        with st.spinner("Querying data..."):
            # Generate embeddings for the data
            data_texts = [str(data_point) for data_point in st.session_state.query_result]  # Convert your data points to strings
            data_embeddings = get_embeddings(data_texts)

            # Generate embedding for the user query
            query_embedding = get_embeddings([result_query_prompt])[0]

            # Find the most relevant data to the query
            relevant_data_idx = find_most_relevant_data(data_embeddings, query_embedding)

            # Use the relevant data to query with OpenAI
            answer, python_code = query_result_data_with_openai(result_query_prompt, st.session_state.query_result.iloc[relevant_data_idx])
            
            # Display the textual answer
            if answer:
                st.write(answer)
            
            # Execute and display the Python code
            if python_code:
                exec(python_code, globals())
                st.pyplot()




#-----------------------------------------------------------


