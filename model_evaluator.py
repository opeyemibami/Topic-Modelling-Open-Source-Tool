from collections import Counter
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from pathlib import Path
from wordcloud import WordCloud, STOPWORDS
import PyPDF2
import streamlit as st

from nltk.corpus import stopwords
stop_words = list(set(stopwords.words('english')))
stop_words.extend(['lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 
'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do',
'done', 'try', 'many','from', 'subject', 're', 'edu','some', 'nice', 'thank',
'think', 'see', 'rather', 'easy', 'easily', 'lot', 'line', 'even', 'also', 'may', 'take', 'come'])

######## chart paths creation ###############
chart1_path = Path.cwd().joinpath('chart1.pdf')
chart2_path = Path.cwd().joinpath('chart2.pdf')
chart3_path = Path.cwd().joinpath('chart3.pdf')
chart_path = Path.cwd().joinpath('chart.pdf')

def del_charts():
    """
    funtion delete charts after downloaded

    """
    charts_list = [chart1_path,chart2_path,chart3_path,chart_path]
    for chart in charts_list:
        chart.unlink()

def get_model_results(corpus, texts,ldamodel=None):
    """funtion extract model result such as topics, percentage distribution and return it as pandas dataframe
    in: corpus : encoded features
    in: text : main text
    in: ladmodel: the trained model
    out: datafram
    """
    
    topics_df = pd.DataFrame()

    # Exrract the main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Percentage Contribution and Topic Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                topics_df = topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    topics_df.columns = ['Dominant_Topic_Number', 'Percentage_Contribution', 'Topic_Keywords']

    # concat original text to topics_df
    contents = pd.Series(texts)
    topics_df = pd.concat([topics_df, contents], axis=1)
    topics_df.columns = ['Dominant_Topic_Number', 'Percentage_Contribution', 'Topic_Keywords','Text']
    topics_df = topics_df[["Text","Topic_Keywords","Dominant_Topic_Number","Percentage_Contribution"]]
    return(topics_df)


cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'


def vis_distribution(n_topics,topics_df):
    """
    funtion visualize topic distribution and export as pdf
    in: n_topics : int -> number of topics
    in: topics_df : dataframe formed containing model results 
    
    """
    if n_topics % 2 == 0 and n_topics<=6:
        fig, axes = plt.subplots(2,n_topics//2,figsize=(16,14), dpi=160, sharex=True, sharey=True)
    elif n_topics % 2 != 0 and n_topics<=6:
        fig, axes = plt.subplots(2,(n_topics+1)//2,figsize=(16,14), dpi=160, sharex=True, sharey=True)
    elif n_topics % 2 == 0 and n_topics>6:
        fig, axes = plt.subplots(3,n_topics//2,figsize=(16,14), dpi=160, sharex=True, sharey=True)
    elif n_topics % 2 != 0 and n_topics>6:
        fig, axes = plt.subplots(3,(n_topics+1)//2,figsize=(16,14), dpi=160, sharex=True, sharey=True)


    for i, ax in enumerate(axes.flatten()):  
        if i == n_topics:
            break
        else:  
            df_dominant_topic_sub = topics_df.loc[topics_df.Dominant_Topic_Number == i, :]
            doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
            ax.hist(doc_lens, bins = 1000, color=cols[i])
            ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
            sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
            ax.set(xlim=(0, 1000), xlabel='Document Word Count')
            ax.set_ylabel('Number of Documents', color=cols[i])
            ax.set_title('Topic: '+str(i), fontdict=dict(size=16, color=cols[i]))

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.xticks(np.linspace(0,1000,9))
    fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)

    
    with PdfPages(chart1_path) as export_pdf:
        export_pdf.savefig()
        # plt.close()
    st.pyplot() 
    

def vis_word_cloud(n_topics,lda_model):
    """
    funtion visualize topics dominat words using word cloud and export as pdf
    in: n_topics : int -> number of topics
    in: lda_model : trained model 
    
    """

    cloud = WordCloud(stopwords=stop_words,
                background_color='white',
                width=2500,
                height=1800,
                max_words=10,
                colormap='tab10',
                color_func=lambda *args, **kwargs: cols[i],
                prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False)

    if n_topics % 2 == 0 and n_topics<=6:
        fig, axes = plt.subplots(2,n_topics//2,figsize=(16,14), dpi=160, sharex=True, sharey=True)
    elif n_topics % 2 != 0 and n_topics<=6:
        fig, axes = plt.subplots(2,(n_topics+1)//2,figsize=(16,14), dpi=160, sharex=True, sharey=True)
    elif n_topics % 2 == 0 and n_topics>6:
        fig, axes = plt.subplots(3,n_topics//2,figsize=(16,14), dpi=160, sharex=True, sharey=True)
    elif n_topics % 2 != 0 and n_topics>6:
        fig, axes = plt.subplots(3,(n_topics+1)//2,figsize=(16,14), dpi=160, sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        if i == n_topics:
            break
        else:
            fig.add_subplot(ax)
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
            plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    fig.suptitle('Topics Words Cloud', fontsize=22)
    
     #export chart as pdf
    with PdfPages(chart2_path) as export_pdf:
        export_pdf.savefig()
        

    st.pyplot() 
   

def vis_count_n_weight(n_topics, lda_model,clean_text):
    """
    funtion visualize word count and importance (weight) and export as pdf
    in: n_topics : int -> number of topics
    in: lda_model : trained model 
    in: clean_text : cleaned text
    
    """
    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in clean_text for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

    # Plot Word Count and Weights of Topic Keywords
    if n_topics % 2 == 0 and n_topics<=6:
        fig, axes = plt.subplots(2,n_topics//2,figsize=(16,14), dpi=160, sharex=True, sharey=True)
    elif n_topics % 2 != 0 and n_topics<=6:
        fig, axes = plt.subplots(2,(n_topics+1)//2,figsize=(16,14), dpi=160, sharex=True, sharey=True)
    elif n_topics % 2 == 0 and n_topics>6:
        fig, axes = plt.subplots(3,n_topics//2,figsize=(16,14), dpi=160, sharex=True, sharey=True)
    elif n_topics % 2 != 0 and n_topics>6:
        fig, axes = plt.subplots(3,(n_topics+1)//2,figsize=(16,14), dpi=160, sharex=True, sharey=True)

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        if i == n_topics:
            break
        else:
            ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
            ax_twin = ax.twinx()
            ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
            ax.set_ylabel('Word Count', color=cols[i])
            ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
            ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
            ax.tick_params(axis='y', left=False)
            ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
            ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)    
    fig.suptitle('Word Count and Importance of Topic Keywords\n ', fontsize=22, y=1)   
    
     #export chart as pdf
    with PdfPages(chart3_path) as export_pdf:
        export_pdf.savefig()
        

    st.pyplot() 


def generate_chart():
    """
    funtion generate final chart by combining exported charts 

    out: pdf1Reader : an oped pdf file 
    
    """
    
    pdf1File = open(chart1_path, 'rb')
    pdf2File = open(chart2_path, 'rb')
    pdf3File = open(chart3_path, 'rb')
    
    # Read the opened files 
    pdf1Reader = PyPDF2.PdfFileReader(pdf1File)
    pdf2Reader = PyPDF2.PdfFileReader(pdf2File)
    pdf3Reader = PyPDF2.PdfFileReader(pdf3File)

    # Create a new PdfFileWriter object which represents a blank PDF document
    pdfWriter = PyPDF2.PdfFileWriter()

    # Loop through all the pagenumbers for the first document
    for pageNum in range(pdf1Reader.numPages):
        pageObj = pdf1Reader.getPage(pageNum)
        pdfWriter.addPage(pageObj)
    # Loop through all the pagenumbers for the second document
    for pageNum in range(pdf2Reader.numPages):
        pageObj = pdf2Reader.getPage(pageNum)
        pdfWriter.addPage(pageObj)
    # Loop through all the pagenumbers for the third document
    for pageNum in range(pdf3Reader.numPages):
        pageObj = pdf3Reader.getPage(pageNum)
        pdfWriter.addPage(pageObj)

    # write them into the a new document
    pdfOutputFile = open(chart_path, 'wb')
    pdfWriter.write(pdfOutputFile)

    # Close all the files - Created as well as opened
    pdfOutputFile.close()
    pdf1File.close()
    pdf2File.close()
    pdf3File.close()
    return pdf1Reader
        

        