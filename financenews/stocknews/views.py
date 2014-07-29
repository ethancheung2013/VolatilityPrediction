# -*- coding: utf-8 -*-
from django.shortcuts import render_to_response
from django.http import HttpResponse
from stocknews.models import NewsContent
import cPickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import ipdb
import numpy as np
import pandas as pd

def cleanurl(url, W_corr):
    '''
        Description:
            Splits url to domain
        Parameters:
            Receives url to be cleaned, dataframe to merge into
    '''
    source = url.split('.')[1]
    if source == 'foxnews':
        W_corr['foxnews'] = 1
        W_corr['html'] = 0
    elif source == 'html':
        W_corr['foxnews'] = 0
        W_corr['html'] = 1
    else:
        W_corr['foxnews'] = 0
        W_corr['html'] = 0
    return W_corr

def calc_topic(W_corr, W):
    '''
        Description:
            Calculates the NMF topic for the document
        Parameters:
            Receives a dataframe, outputs the predicted category
    '''

    topics_dict = {4: "WorldNews", 0: "Corporate", 2: "Housing", 9: "ConsumerSpending", 8: "CorporateEarnings", \
                      7: "Energy", 6: "Employment", 5: "Government", 1: "Auto", 3: "General"}
    topic = np.argsort(W_corr[::-1]).iloc[:,W.shape[1]-1:W.shape[1]]
    pred_category = topics_dict.get(topic.values[0][0])
    return pred_category

# Create your views here.

def hello(request):
    print 'hi from inside hello'

def index(request):
    def clean_data(text):
        return ''.join([i if ord(i) < 128 else ' ' for i in text])

    # Baseline model: kkm_lr_tfi_model.pkl
    with open ('/Users/ethancheung/Documents/zipfianacademy/FoxScraper/lr_lin_n_clf.pkl', 'rb') as fid:
        n_clf, lin_clf, lr_clf, rf_clf, tfidf, nb_classifer = cPickle.load(fid)

    n_samples = 2000
    n_features = 1000
    n_topics = 10
    n_top_words = 20

    tokenizer = RegexpTokenizer(r'\w+')
    top_10 = NewsContent.objects.all()[:50]

    sent_arry = []
    for idx, eDoc in enumerate(top_10):
        # classify the sentiment
        title = eDoc.title
        newdict = {}
        for i in tokenizer.tokenize(title):
            newdict[i] = True
        print 'finished tokenizing'
        sentiment = nb_classifer.classify(newdict)
        content =  ''.join([i if ord(i) < 128 else ' ' for i in eDoc.content])

        clfv = tfidf.transform([content])

        # determine the topic
        W = n_clf.transform(clfv)
        W_corr = pd.DataFrame(W)
        pred_category = calc_topic(W_corr, W)

        url = eDoc.url
        date1 = eDoc.date
        W_corr['Date'] = pd.to_datetime(date1).dayofweek
        sentiment_dict = {'pos': 0, 'neg': 1}
        W_corr['Sentiment'] = sentiment_dict.get(sentiment)

        W_corr = cleanurl(url, W_corr)

        y_pred = lr_clf.predict(W_corr)
        vol_pred1 = lin_clf.predict(W_corr)
        vol_pred2 = rf_clf.predict(W_corr)


        sent_arry.append({"title": title, "content": content, "url": url, "category": pred_category, "sentiment": sentiment, "has_volatility": y_pred, "volatility_lr": vol_pred1, "volatility_rf": vol_pred2})

    return render_to_response("stocknews/index2.html", { "news" : sent_arry })




