import streamlit as st 
import pandas as pd  


def select_text_feature(df):
    text_col = st.selectbox('Select the text column',(list(df.columns)))
    return text_col

def rename_column(df,text_col):
    df = df.rename(columns={text_col:'text'})
    df = pd.DataFrame(df['text'])
    st.write(df.head())
    return df

def get_input(is_batch=False,text_column = "text"):
    if is_batch:
        uploaded_file = st.file_uploader("Choose a CSV file to upload", type="csv")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(df.head())
            return df
        else:
            st.write('Kindly upload a csv file')

    else: 
        txt = st.text_area('Text to analyze','Paste the text to analyze here')
        df = pd.DataFrame(data=[txt],columns=[text_column])
        st.write(df.head())
        return df


