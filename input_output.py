import streamlit as st 
import pandas as pd  


def select_text_feature(df):
    text_col = st.selectbox('Select the text column',(list(df.columns)))
    df = pd.DataFrame(df[text_col])
    st.write(df.head())
    return df,text_col

# def rename_column(df,text_col):
#     columns = df.columns
#     if text_col != 'text' and 'text' not in columns:
#         df = df.rename(columns={text_col:'text'})
#         st.success(f"'{text_col}' column successfully renamed to 'text'")
#         # df = pd.DataFrame(df['text'])
#         st.write(df.head())
#     elif text_col != 'text' and 'text' in columns:
#         st.write(f"Warning!!! {text_col} column cannot be renamed. The default text column will be used if not changed")
#     elif text_col == 'text':
#         st.write("Warning!!! 'text' column already exist, All column names has to be unique. The default text column will be used if not changed")
        
#     return df

# @st.cache
def get_input(ss_text,is_batch=False,text_column = "text"):
    if is_batch:
        uploaded_file = st.file_uploader("Choose a CSV file to upload", type="csv")

        if uploaded_file is not None:
            st.success('Filel successfuly uploaded')
            df = pd.read_csv(uploaded_file)
            ############################################### top 5 header ###################################
            return df,ss_text
        else:
            st.write('Kindly upload a csv file')
            return pd.DataFrame(),ss_text

    else: 
        ss_text = st.text_area('Text to analyze',ss_text)
        df = pd.DataFrame(data=[ss_text],columns=[text_column])
        ############################################### top rows header #######################################
        return df,ss_text


