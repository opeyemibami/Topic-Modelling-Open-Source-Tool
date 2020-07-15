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

st.write("""
# Topic Modelling Open-Source Tool 
""")

############ configuration ###################

st.sidebar.header("""
Instance Configuration 
\n Use this side panel to configure the modelling space 
""")

data_input_mthd = st.sidebar.radio("Select Data Input Method",('Copy-Paste text', 'Upload a CSV file'))
clean_data_opt = st.sidebar.radio("Clean the Data or Use Raw Data",('Use Raw Data', 'Clean the Data'))
st.sidebar.text('Feature extraction section')
normalization_mthd = st.sidebar.radio("Select a text normalization method",('None','Lemmatization', 'Stemming'))
encoding_mthd = st.sidebar.selectbox('Select a feature extraction method',(['None','BOW with Term Frequency','BOW with TF-IDF']))
n_of_topics = st.sidebar.number_input('Expected Number of Topics',min_value=1,value=2,step =1)
# top_k = st.sidebar.number_input('Expected Number of Top Words in each Topics',min_value=1,value=2,step =1)
model = st.sidebar.radio("Select Model Type",('Latent Dirichlet Allocation', 'Non-Negative Matrix Factorization'))


ss = SessionState.get(output_df = pd.DataFrame(), 
df_raw = pd.DataFrame(),
encoded_data = None,
_feature_names=[],
_model=None,
n_topics =n_of_topics, 
text_col='text',
# is_txt_col_selected = False,
is_file_uploaded=False,

id2word = None, 
corpus= None,

to_clean_data = False,
to_encode = False,
to_train = False,
to_evaluate = False,
to_visualize = False,
to_download_report = False,

# is_batch_process= False,
# button_rename = False, 
# button_clean_data=False, 
df = pd.DataFrame(),
txt = 'Paste the text to analyze here',
default_txt = 'Paste the text to analyze here',
clean_text = None,
ldamodel = None,
topics_df = None,
chart_pdf = None)


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
ss.df,ss.txt  = check_input_method(data_input_mthd)

if ss.text_col != ss.default_txt:
    ss.to_clean_data = True

ss.df_raw = ss.df.copy()

if ss.is_file_uploaded:
    ss.df,ss.text_col = io.select_text_feature(ss.df)
    ss.to_clean_data = True


# clean data #######
if ss.to_clean_data:
    if clean_data_opt=='Use Raw Data':
        st.write('################ Using Raw data Header ###############')
        st.write(ss.df_raw.head())
        if ss.text_col != ss.default_txt:
            ss.to_encode = True
    else:
        st.write('################### Using Clean Header ############################')
        ss.df = pp.clean_data(ss.df,feature=ss.text_col)
        st.success('Data cleaning successfuly done')
        ss.to_encode = True
        st.write(ss.df.head())

# Encoding ##############

if ss.to_encode and encoding_mthd !='None':
    st.write('######## Data Encoding Section #####################')
    st.write('Select an ecoding method in the side panel')
    
    if encoding_mthd=='BOW with Term Frequency':
        st.write('######## Using Term Frequency Endcoding #####################')
        ss.id2word, ss.corpus,ss.clean_text=pp.extract_features(ss.df,feature=ss.text_col,normalization_mthd=normalization_mthd,mode='Term Frequency')
        # st.write(ss.corpus[0])
        # ss.encoded_data,ss._feature_names=pp.encoder(ss.df,feature=ss.text_col,mode='freq')
        st.success('Data Encoding Successfully done with Term Frequency')
        ss.to_train = True

    elif encoding_mthd=='BOW with TF-IDF':
        st.write('######## Using Term Frequency - Inverse Term Frequency Endcoding #####################')
        ss.id2word, ss.corpus,ss.clean_text=pp.extract_features(ss.df,feature=ss.text_col,normalization_mthd=normalization_mthd,mode='Term Frequency')
        # ss.encoded_data,ss._feature_names=pp.encoder(ss.df,feature=ss.text_col,mode='tfidf')
        st.success('Data Encoding Successfully done with Term Frequency - Inverse Term Frequency')
        ss.to_train = True
else:
    st.write('Select an ecoding method in the side panel')

################### Training ###########################

if ss.to_train:
    st.write('######## Training section header #####################')
    button_train = st.button('Train Model')
    if button_train:
        if model== 'Latent Dirichlet Allocation':
            ss._model = LDA.lda_train(ss.corpus,ss.id2word,number_of_topics=n_of_topics)
            ss.output_df= ss.df.copy()
            st.success('Training successful!!!')
            # st.write(ss.df.head())
            ss.to_evaluate = True
        elif model=='Non-Negative Matrix Factorization':
           st.write('Non-Negative Matrix Factorization is yet to be implemented')


################### Model Evaluation  ###########################
if ss.to_evaluate:
    st.write('######## Model Evaluation Section #####################')
    button_eva = st.button('Evaluate Model')
    if button_eva:
        ss.topics_df = mv.get_model_results(corpus=ss.corpus, texts = ss.clean_text,ldamodel=ss._model)

        # Formatting 
        ss.topics_df = ss.topics_df.reset_index()
        ss.topics_df.columns = ["Document_No", "Text","Topic_Keywords","Dominant_Topic_Number","Percentage_Contribution"]
        st.write('First 10 Rows of the Model Output')
        st.write(ss.topics_df.head(10))
        ss.to_visualize = True


################### Model Evaluation with Visualization ###########################
if ss.to_visualize:
    st.write('######## Model Evaluation Section #####################')
    button_vis = st.button('Evaluate with Visuals')
    if button_vis:
        mv.vis_distribution(n_of_topics,ss.topics_df)
        mv.vis_word_cloud(n_of_topics,ss._model)
        mv.vis_count_n_weight(n_of_topics, ss._model,ss.clean_text)
        ss.to_download_report = True


################### Downloading Section ###########################
if ss.to_download_report:
    st.write('######## Download result #####################')
    button_download = st.button('Generate Report Sheet')
    if button_download and Path(Path.cwd().joinpath('chart1.pdf')).is_file():
        st.success('Report successfuly generated')
        ss.chart_pdf = mv.generate_chart()
        st.markdown(get_table_download_link(ss.topics_df), unsafe_allow_html=True)
        st.markdown(get_chat_download_link(ss.chart_pdf), unsafe_allow_html=True)
        mv.del_charts()
    elif button_download and not Path(Path.cwd().joinpath('chart1.pdf')).is_file():
        st.write("Kindly click the 'Evaluate with Visuals' button above to first create reports ")
        
