import pandas as pd
import streamlit as st 
import SessionState
import input_output as io
import preprocessor as pp
import model_evaluator as mv
import lda as LDA 
import base64
from io import BytesIO
from pathlib import Path
import time


def display_app_header(main_txt,sub_txt,is_sidebar = False):
    # Open Source Topic Modelling Web App
    # Build and Evaluate a Topic Model without coding!!!
    html_temp = f"""
    <div style = "background.color:#054029  ; padding:15px">
    <h2 style = "color:white; text_align:center;"> {main_txt} </h2>
    <p style = "color:white; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)

def display_side_panel_header(txt):
    st.sidebar.markdown(f'## {txt}')

#Main panel title
display_app_header(main_txt='Topic Modelling Open Source Tool',sub_txt='Build and Evaluate a Topic Model without coding!!!')
#Side panel title
display_app_header(main_txt='Instance Configuration ',sub_txt='Use this panel to configure the modelling space and complete corresponding actions after each step in the main panel',is_sidebar=True)




display_side_panel_header(txt='Step 1:')
data_input_mthd = st.sidebar.radio("Select Data Input Method",('Copy-Paste text', 'Upload a CSV file'))

display_side_panel_header(txt='Step 2:')
clean_data_opt = st.sidebar.radio("Clean the Data or Use Raw Data",('Use Raw Data', 'Clean the Data'))

display_side_panel_header(txt='Step 3:')
normalization_mthd = st.sidebar.radio("Select a text normalization method",('None','Lemmatization', 'Stemming'))
encoding_mthd = st.sidebar.selectbox('Select a feature extraction method',(['None','BOW with Term Frequency','BOW with TF-IDF']))

display_side_panel_header(txt='Step 4:')
model = st.sidebar.radio("Select Model Type and set Hyperparameters. Default parameters actually work fine",('Latent Dirichlet Allocation', 'Non-Negative Matrix Factorization'))
n_of_topics = st.sidebar.number_input('Expected Number of Topics',min_value=1,value=5,step =1)
update_every= st.sidebar.slider('update_every (0: batch learning, 1: online iterative learning.)', 0,1,1)
chunksize= st.sidebar.slider('chunksize (Number of documents to be used in each training chunk))', 10,20,10)
passes= st.sidebar.slider('passes (Number of passes through the corpus during training))', 10,20,10)
alpha= st.sidebar.selectbox('Alpha',(['symmetric','auto']))
iterations=st.sidebar.number_input('Number of Iteration',min_value=50,max_value=500,value=100,step =1)

#How to contribute 
display_app_header(main_txt='How to Contribute',sub_txt='Send Pull Request to Project Repository on Github (Topic-Modelling-Open-Source-Tool)',is_sidebar=True)

# Project Version:
display_app_header(main_txt='Project Version',sub_txt=' 0.1.0 c2020',is_sidebar=True)
#About Aauthor
display_app_header(main_txt='About Author',sub_txt='Opeyemi Bamigbade (Yhemmy) is a Data Scientist and ML-Engineer at Data Science Nigeria. Twitter:@opeyemiami    Email:bamigbadeopeyemi@gmail.com',is_sidebar=True)

#session state 
ss = SessionState.get(output_df = pd.DataFrame(), 
df_raw = pd.DataFrame(),
_model=None,
text_col='text',
is_file_uploaded=False,

id2word = None, 
corpus= None,

is_valid_text_feat = False,
to_clean_data = False,
to_encode = False,
to_train = False,
to_evaluate = False,
to_visualize = False,
to_download_report = False,


df = pd.DataFrame(),
txt = 'Paste the text to analyze here',
default_txt = 'Paste the text to analyze here',
clean_text = None,
ldamodel = None,
topics_df = None,
chart_pdf = None)



def display_header(header):
     
    #view clean data
    html_temp = f"""
    <div style = "background.color:#054029; padding:10px">
    <h4 style = "color:white;text_align:center;"> {header} </h5>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

def space_header():
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


st.cache()
def check_input_method(data_input_mthd):
    if data_input_mthd=='Copy-Paste text':
        df,ss.txt = io.get_input(ss_text= ss.txt)


    else:
        df,ss.txt= io.get_input(is_batch=True,ss_text= ss.txt)
        if df.shape[0]>0:
            # ss.is_batch_process = True
            ss.is_file_uploaded = True
    
    return df,ss.txt
    

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="Report.csv" >Download csv file</a>' 

    return href

def get_chat_download_link(chart_pdf):
    # base64_pdf = None
    with open("chart.pdf", "rb") as pdf_file:
        base64_pdf = base64.b64encode(pdf_file.read()).decode()
    href = f'<a href="data:file/pdf;base64,{base64_pdf }" download="Charts.pdf" >Download charts file</a>' 

    return href



############ APP Logical Flow ###############

space_header()
ss.df,ss.txt  = check_input_method(data_input_mthd)

if ss.text_col != ss.default_txt:
    ss.to_clean_data = True

ss.df_raw = ss.df.copy()

if ss.is_file_uploaded:
    ss.df,ss.text_col = io.select_text_feature(ss.df)
    if ss.df[ss.text_col].dtype =='O':
        ss.to_clean_data = True
        ss.is_valid_text_feat =True
    else:
        st.warning('select a valid text column')
        ss.to_clean_data = False

# clean data #######
if ss.to_clean_data:
    if clean_data_opt=='Use Raw Data':
        display_header(header = 'Using Raw data')  #Raw data header
        space_header()
        st.write(ss.df_raw.head())
        if ss.text_col != ss.default_txt:
            ss.to_encode = True
    else:
        display_header(header = 'Using Clean Data')  #Clean data header
        space_header()
        ss.df = pp.clean_data(ss.df,feature=ss.text_col)
        st.success('Data cleaning successfuly done')
        ss.to_encode = True
        st.write(ss.df.head())

# Encoding ##############

if ss.to_encode and encoding_mthd !='None':
    display_header(header = 'Data Encoding Section ')  #Encoding header
    space_header()
    
    if encoding_mthd=='BOW with Term Frequency':
        ss.id2word, ss.corpus,ss.clean_text=pp.extract_features(ss.df,feature=ss.text_col,normalization_mthd=normalization_mthd,mode='Term Frequency')
        st.success('Data Encoding Successfully done with Term Frequency')
        ss.to_train = True

    elif encoding_mthd=='BOW with TF-IDF':
        ss.id2word, ss.corpus,ss.clean_text=pp.extract_features(ss.df,feature=ss.text_col,normalization_mthd=normalization_mthd,mode='Term Frequency')
        st.success('Data Encoding Successfully done with Term Frequency - Inverse Term Frequency')
        ss.to_train = True
       
elif ss.to_encode and ss.is_valid_text_feat and encoding_mthd =='None':
    st.info('Select an ecoding method in the side panel')

################### Training ###########################

if ss.to_train:
    display_header(header = 'Model Training Section  ')  #Model Training header
    space_header()
    button_train = st.button('Train Model')
    if button_train:
        if model== 'Latent Dirichlet Allocation':
            ss._model = LDA.lda_train(ss.corpus,ss.id2word,update_every,chunksize,passes,alpha,iterations,number_of_topics=n_of_topics)
            ss.output_df= ss.df.copy()
            st.success('Training completed!!!')
            ss.to_evaluate = True
        elif model=='Non-Negative Matrix Factorization':
           st.error('Non-Negative Matrix Factorization is yet to be implemented, Select LDA')


################### Model Evaluation  ###########################
if ss.to_evaluate:
    display_header(header = 'Model Evaluation Section')  #Model Evaluation header
    space_header()
    button_eva = st.button('Evaluate Model')
    if button_eva:
        ss.topics_df = mv.get_model_results(corpus=ss.corpus, texts = ss.clean_text,ldamodel=ss._model)

        # Formatting 
        ss.topics_df = ss.topics_df.reset_index()
        ss.topics_df.columns = ["Document_No", "Text","Topic_Keywords","Dominant_Topic_Number","Percentage_Contribution"]
        st.info('First few Rows of the Model Output')
        st.write(ss.topics_df.head(10))
        ss.to_visualize = True


################### Model Evaluation with Visualization ###########################
if ss.to_visualize:
    display_header(header = 'Topics Visualization')  #Topics Visualization header
    space_header()
    button_vis = st.button('Evaluate with Visuals')
    if button_vis:
        mv.vis_distribution(n_of_topics,ss.topics_df)
        mv.vis_word_cloud(n_of_topics,ss._model)
        mv.vis_count_n_weight(n_of_topics, ss._model,ss.clean_text)
        st.success('Visualization completed!!!')
        ss.to_download_report = True


################### Downloading Section ###########################
if ss.to_download_report:
    display_header(header = 'Download Report Section')  #Report Section header
    space_header()
    button_download = st.button('Generate Report Sheet')
    if button_download and Path(Path.cwd().joinpath('chart1.pdf')).is_file():
        ss.chart_pdf = mv.generate_chart()
        st.success('Report successfuly generated, click the links below to download.')
        st.markdown(get_table_download_link(ss.topics_df), unsafe_allow_html=True)
        st.markdown(get_chat_download_link(ss.chart_pdf), unsafe_allow_html=True)
        mv.del_charts()
        st.balloons()
    elif button_download and not Path(Path.cwd().joinpath('chart1.pdf')).is_file():
        st.info("Kindly click the 'Evaluate with Visuals' button above to first create reports ")
    