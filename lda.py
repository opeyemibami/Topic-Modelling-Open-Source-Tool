import pandas as pd
import gensim
def lda_train(corpus,id2word,update_every,chunksize,passes,alpha,iterations,number_of_topics=5):
    """funtion create an instance of lda model using gensim and training occur
    in: corpus,id2word and other hyperparameters
    out: trained lda model
    """
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=number_of_topics, 
                                           random_state=2020,
                                           update_every=update_every,
                                           chunksize=chunksize,
                                           passes=passes,
                                           alpha=alpha,
                                           iterations=iterations,
                                           per_word_topics=True)
    return lda_model


