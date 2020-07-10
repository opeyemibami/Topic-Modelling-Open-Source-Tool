import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image


# Load  model a 
model = joblib.load(open("model-v1.joblib","rb"))

def data_preprocessor(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    df.wine_type = df.wine_type.map({'white':0, 'red':1})
    return df

def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(data = data,columns = ['Percentage'],index = ['Low','Ave','High'])
    ax = grad_percentage.plot(kind='barh', figsize=(7, 4), color='#722f37', zorder=10, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel(" Percentage(%) Confidence Level", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Wine Quality", labelpad=10, weight='bold', size=12)
    ax.set_title('Prediction Confidence Level ', fontdict=None, loc='center', pad=None, weight='bold')

    st.pyplot()
    return

st.write("""
# Wine Quality Prediction ML Web-App 
This app predicts the ** Quality of Wine **  using **wine features** input via the **side panel** 
""")

#read in wine image and render with streamlit
image = Image.open('wine_image.png')
st.image(image, caption='wine company',use_column_width=True)

st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe

    """
    wine_type = st.sidebar.selectbox("Select Wine type",("white", "red"))
    fixed_acidity = st.sidebar.slider('fixed acidity', 3.8, 15.9, 7.0)
    volatile_acidity = st.sidebar.slider('volatile acidity', 0.08, 1.58, 0.4)
    citric_acid  = st.sidebar.slider('citric acid', 0.0, 1.66, 0.3)
    residual_sugar  = st.sidebar.slider('residual_sugar', 0.6, 65.8, 10.4)
    chlorides  = st.sidebar.slider('chlorides', 0.009, 0.611, 0.211)
    free_sulfur_dioxide = st.sidebar.slider('free sulfur dioxide', 1, 289, 200)
    total_sulfur_dioxide = st.sidebar.slider('total sulfur dioxide', 6, 440, 150)
    density = st.sidebar.slider('density', 0.98, 1.03, 1.0)
    pH = st.sidebar.slider('pH', 2.72, 4.01, 3.0)
    sulphates = st.sidebar.slider('sulphates', 0.22, 2.0, 1.0)
    alcohol = st.sidebar.slider('alcohol', 8.0, 14.9, 13.4)
    
    features = {'wine_type': wine_type,
            'fixed_acidity': fixed_acidity,
            'volatile_acidity': volatile_acidity,
            'citric_acid': citric_acid,
            'residual_sugar': residual_sugar,
            'chlorides': chlorides,
            'free_sulfur_dioxide': free_sulfur_dioxide,
            'total_sulfur_dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol
            }
    data = pd.DataFrame(features,index=[0])

    return data

user_input_df = get_user_input()
processed_user_input = data_preprocessor(user_input_df)

st.subheader('User Input parameters')
st.write(user_input_df)

prediction = model.predict(processed_user_input)
prediction_proba = model.predict_proba(processed_user_input)

visualize_confidence_level(prediction_proba)