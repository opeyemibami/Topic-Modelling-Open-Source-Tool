import streamlit as st 
# from streamlit import SessionState
import input_output as io
import preprocessor as pp
# import lda as LDA 


st.write("""
# Topic Modelling Open-Source Tool 
""")

############ configuration ###################

st.sidebar.header("""
Instance Configuration 
\n Use this side panel to configure the modelling space 
""")

data_input_mthd = st.sidebar.radio("Select Data Input Method",('Copy-Paste text', 'Upload a CSV file'))
n_of_topics = st.sidebar.number_input('Expected Number of Topics',min_value=1,value=2,step =1)
encoding_mthd = st.sidebar.selectbox('Select an Encoding Method',(['Binary', 'Word Count','Term Frequency','TF-IDF']))
model = st.sidebar.radio("Select Model Type",('Latent Dirichlet Allocation', 'Non-Negative Matrix Factorization'))







st.sidebar.button('Say hello')
st.sidebar.checkbox('I agree')
genre = st.sidebar.radio("What's your favorite movie genre",('Comedy', 'Drama', 'Documentary'))
option = st.sidebar.selectbox('How would you like to be contacted?',(['Email', 'Home phone', 'Mobile phone']))
options = st.sidebar.multiselect('What are your favorite colors',['Green', 'Yellow', 'Red', 'Blue'],['Yellow', 'Red'])
# st.text_input('Movie title', 'Life of Brian')
number = st.sidebar.number_input('Insert a number')
txt = st.sidebar.text_area('Text to analyze', '''
It was the best of times, it was the worst of times, it was
the age of wisdom, it was the age of foolishness, it was
the epoch of belief, it was the epoch of incredulity, it
was the season of Light, it was the season of Darkness, it
was the spring of hope, it was the winter of despair, (...)
''')
# uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")


st.cache()
def check_input_method(data_input_mthd):
    if data_input_mthd=='Copy-Paste text':
        df = io.get_input()
    else:
        df = io.get_input(is_batch=True)
       
    return df
    

st.cache()
def batch_process(df):
    text_col = io.select_text_feature(df)
    if st.button('Raname Text Column'):
        df = io.rename_column(df,text_col)
    else:
        st.write("Make sure to Click the 'Rename Text Column button' above")
    return df

##### APP Logical Flow #########
df = check_input_method(data_input_mthd)
if data_input_mthd=='Upload a CSV file' and df is not None:
    df = batch_process(df)
st.cache({"data":df})
st.write(df.columns)
if st.button('Clean Data'):
    df = pp.clean_data(df)
else:
    st.write("Ensure to Click the 'Clean Data button' above to get the best result")
