import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai

# Ensure your API key is handled securely and not exposed in your code
openai_api_key = ''
openai.api_key = openai_api_key

st.title('Data Explorer with OpenAI')

# Function to load data
@st.cache(allow_output_mutation=True)
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# Function to answer questions about the data using OpenAI API
def answer_data_question(prompt, df):
    # Convert the DataFrame to a string in a CSV-like format for the prompt
    df_string = df.to_csv(index=False)

    # Set up the conversation with the model for answering questions
    messages = [
        {
            "role": "system",
            "content": "You are a data analyst. Answer the question based on the data provided."
        },
        {
            "role": "user",
            "content": f"{prompt}\n\nInput data:\n{df_string}"
        }
    ]

    try:
        # Use OpenAI to generate a response for answering questions
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Extract the answer directly from the 'content' key of the 'message' dictionary
        answer = response['choices'][0]['message']['content'].strip()
        
        return answer
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return None

# File uploader
uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write(data)  # Display the uploaded data

    # User query input
    user_query = st.text_area("Ask a question about the data:")

    # If a query is entered, use OpenAI API to get the response
    if user_query and st.button('Get Answer'):
        with st.spinner('Getting response from OpenAI...'):
            openai_response = answer_data_question(user_query, data)
            if openai_response:
                st.write('OpenAI response:', openai_response)

# Sample plotting functionality (not integrated with OpenAI response)
st.sidebar.header("Plotting")
plot_choices = st.sidebar.radio("Choose a plot type:", ["Bar", "Count", "Line"])
selected_column = st.sidebar.selectbox("Select column to plot:", data.columns)

if st.sidebar.button('Generate Plot'):
    if plot_choices == "Bar":
        fig, ax = plt.subplots()
        sns.barplot(x=data[selected_column].value_counts().index, y=data[selected_column].value_counts().values)
        st.pyplot(fig)
    elif plot_choices == "Count":
        fig, ax = plt.subplots()
        sns.countplot(x=selected_column, data=data)
        st.pyplot(fig)
    elif plot_choices == "Line":
        # Assuming the selected column is a numeric value for plotting line chart
        fig, ax = plt.subplots()
        sns.lineplot(data=data[selected_column].dropna().reset_index(drop=True))
        st.pyplot(fig)
