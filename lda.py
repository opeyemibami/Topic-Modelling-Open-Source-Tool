from sklearn.decomposition import LatentDirichletAllocation

def lda_train(df,encoded_data,feature_names,number_of_topics=5,top_k=10):
    top_n_words = []
    topic_top_words = []
    LDA = LatentDirichletAllocation(random_state=2020,n_components=number_of_topics)
    topics = LDA.fit_transform(encoded_data).argmax(axis=1)

    for topic in LDA.components_:
        top_n_words.append([feature_names[i] for i in topic.argsort()[-top_k:]])

    for topic in topics:
        topic_top_words.append(top_n_words[topic])

    return topics,topic_top_words
