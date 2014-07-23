# -*- coding: utf-8 -*-
from django.shortcuts import render_to_response
from django.http import HttpResponse
from stocknews.models import NewsContent
import cPickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import ipdb

# from sentimentmodel import *

# Create your views here.
def news(request):

    return render_to_response("stocknews/base_news.html", { "news" : NewsContent.objects.all() })

def index(request):

    def clean_data(text):
        return ''.join([i if ord(i) < 128 else ' ' for i in text])

    # Baseline model: kkm_lr_tfi_model.pkl
    with open ('/Users/ethancheung/Documents/zipfianacademy/FoxScraper/km_lrnmf_tfi_model.pkl', 'rb') as fid:
        logclf_loaded, tfclf_loaded, nb_classifer = cPickle.load(fid)

    n_samples = 2000
    n_features = 1000
    n_topics = 10
    n_top_words = 20

    # # just tranform since the pickled version has already fitted
    # clfv = tfclf_loaded.transform(data)

    # y_pred = logclf_loaded.predict(clfv)

    # assumes data is array, newdict needs to be reinitialized each time
    # data is passed from urls.py
    # newdict = {}
    # for i in tokenizer.tokenize(data[0]):
    #     newdict[i] = True
    # sentiment = nb_classifer.classify(newdict)
    # print sentiment

    # print y_pred
    tokenizer = RegexpTokenizer(r'\w+')
    top_10 = NewsContent.objects.all()[:5]

    sent_arry = []
    for eDoc in top_10:
        # classify the sentiment
        title           = eDoc.title
        newdict         = {}
        for i in tokenizer.tokenize(title):
            newdict[i] = True
        sentiment = nb_classifer.classify(newdict)
        content = eDoc.content
        # classify the catogory
        # content         = clean_data(eDoc.content)
        clfv            = tfclf_loaded.transform([content])
        pred_category   = logclf_loaded.predict(clfv)

        url = eDoc.url

        sent_arry.append({"title": title, "content": content, "url": url, "category": pred_category, "sentiment": sentiment})

    return render_to_response("stocknews/base.html", { "news" : sent_arry })
    # return render_to_response("stocknews/bbb.html", { "news" : top_10 })
    
    # return HttpResponse("Hello shitty world!")



