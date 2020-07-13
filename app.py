import pandas as pd
import streamlit as st 
import SessionState
import input_output as io
import preprocessor as pp
import lda as LDA 


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
encoding_mthd = st.sidebar.selectbox('Select an Encoding Method',(['None','Binary', 'Word Count','Term Frequency','TF-IDF']))
n_of_topics = st.sidebar.number_input('Expected Number of Topics',min_value=1,value=2,step =1)
top_k = st.sidebar.number_input('Expected Number of Top Words in each Topics',min_value=1,value=2,step =1)
model = st.sidebar.radio("Select Model Type",('Latent Dirichlet Allocation', 'Non-Negative Matrix Factorization'))


ss = SessionState.get(output_df = pd.DataFrame(), 
df_raw = pd.DataFrame(),
encoded_data = None,
_feature_names=[],
_model=model,
n_topics =n_of_topics, 
text_col='text',
# is_txt_col_selected = False,
is_file_uploaded=False,

to_clean_data = False,
to_encode = False,
to_train = False,
to_visualize = False,

# is_batch_process= False,
# button_rename = False, 
# button_clean_data=False, 
df = pd.DataFrame(),
txt = 'Paste the text to analyze here',
default_txt = 'Paste the text to analyze here')

# ss = SessionState.get(a=2,b=3)
# st.write(ss.df)



# uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")


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
    

# st.cache()
# def rename_column(df):
#     if ss.button_rename==False:
#         text_col = io.select_text_feature(df)
#         button_rename = st.button('Raname Text Column')
#         if button_rename: 
#             ss.button_rename=True    #session input
#             return io.rename_column(df,text_col)
#         else:
#             st.write("Make sure to Click the 'Rename Text Column button' above")
#     else:
#         return df
    

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
    
    if encoding_mthd=="Binary":
        st.write('######## Using Binary Endcoding #####################')
        ss.encoded_data,ss._feature_names=pp.encoder(ss.df,feature=ss.text_col)
        st.success('Data Encoding Successfully done with Binary')
        ss.to_train = True
    # st.write(ss._feature_names)
    elif encoding_mthd=="Word Count":
        st.write('######## Using Word Count Endcoding #####################')
        ss.encoded_data,ss._feature_names=pp.encoder(ss.df,feature=ss.text_col,mode='count')
        st.success('Data Encoding Successfully done with Word count')
        ss.to_train = True

    elif encoding_mthd=="Term Frequency":
        st.write('######## Using Term Frequency Endcoding #####################')
        ss.encoded_data,ss._feature_names=pp.encoder(ss.df,feature=ss.text_col,mode='freq')
        st.success('Data Encoding Successfully done with Term Frequency')
        ss.to_train = True

    elif encoding_mthd=="TF-IDF":
        st.write('######## Using Term Frequency - Inverse Term Frequency Endcoding #####################')
        ss.encoded_data,ss._feature_names=pp.encoder(ss.df,feature=ss.text_col,mode='tfidf')
        st.success('Data Encoding Successfully done with Term Frequency - Inverse Term Frequency Endcoding')
        ss.to_train = True


################### Training ###########################

if ss.to_train:
    st.write('######## Training section header #####################')
    button_train = st.button('Train Model')
    if button_train:
        ss.df['topics'],ss.df['top_'+ str(top_k) +'_words']=LDA.lda_train(ss.df,ss.encoded_data,ss._feature_names,number_of_topics=n_of_topics,top_k=top_k)
        ss.output_df= ss.df.copy()
        st.success('Training successful!!!')
        st.write(ss.df.head())
        ss.to_visualize = True

################### Visualize ###########################
if ss.to_visualize:
    st.write('######## Visualization Section #####################')
    button_vis = st.button('Visualise')
    if button_vis:
        st.write(ss.output_df.head())

################### Downloading Section ###########################
st.write('######## Download result #####################')
