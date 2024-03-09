import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI
import pandas as pd
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_experimental.agents import create_pandas_dataframe_agent

def delete_images(file_paths):
    for path in file_paths:
        try:
            os.remove(path)
            print(f"Deleted: {path}")
        except OSError as e:
            print(f"Error deleting {path}: {e}")

def Agent_function(df):
    df_columns = df.columns
    print(df_columns)

    #converting list to string, 
    df_columns_text = ', '.join(df_columns)
    print(df_columns_text)


    llm = ChatOpenAI(openai_api_key="")
    # response = llm.invoke("Hello, who are you?")
    # print(response)

    python_repl = PythonREPL()

    repl_tool = Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run,
    )

    query = f"""
    consider we have alreaded loaded pandas DataFrame 'df' in pandas, to plot graphs. By looking at all dataframe and these features: {df_columns_text} ,By looking at the df, By checking data features convert only important, some categorical features to numerical and perform 10 various colourful visualizations .
    for every visualization code, add: " plt.savefig(filename...)"  save the image and tell me where it saved
    """


    agent = create_pandas_dataframe_agent(llm, df, agent_type="openai-tools", verbose=True)
    answer = agent.invoke(
        {
            "input": query
        }
    )
    print(answer)
    
import os
import streamlit as st
from PIL import Image

def get_image_files(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                image_files.append(os.path.join(folder_path, file))
    return image_files

def before_get_image_files():
    folder_path = 'D:\OpenAI LLM'
    image_files = get_image_files(folder_path)
    print("Image files in the folder:")
    return image_files


def Image_streamlit():
    image_files = before_get_image_files()
    st.title("Image Viewer")
    # List of image file paths
    image_paths = image_files
    # Display images
    for image_path in image_paths:
        image = Image.open(image_path)
        st.image(image, caption=image_path, use_column_width=True)
        return image_paths


def main():
    st.title("File Upload, DataFrame Display, and Visualization")

    # File upload section
    st.sidebar.title("Upload File")
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Read the uploaded file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return
        except Exception as e:
            st.error(f"Error: {e}")
            return

        # Display DataFrame
        st.subheader("Uploaded DataFrame")
        st.write(df)

        # Display charts
        st.subheader("Data Visualization")
        print("sending df to agent...")
        Agent_function(df)
        image_paths = Image_streamlit()
        #return image_paths


if __name__ == "__main__":
    image_paths = main()
    #delete_images(image_paths)