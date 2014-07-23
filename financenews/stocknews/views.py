# -*- coding: utf-8 -*-
from django.shortcuts import render_to_response
from django.http import HttpResponse
from stocknews.models import NewsContent
import cPickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import ipdb


# Create your views here.

def hello(request):
    print 'hi from inside hello'

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

    tokenizer = RegexpTokenizer(r'\w+')
    top_10 = NewsContent.objects.all()[:5]

    sent_arry = []
    for eDoc in top_10:
        # classify the sentiment
        title           = eDoc.title
        newdict         = {}
        for i in tokenizer.tokenize(title):
            newdict[i] = True
        print 'finished tokenizing'
        sentiment = nb_classifer.classify(newdict)
        content = eDoc.content
        # classify the catogory
        clfv            = tfclf_loaded.transform([content])
        pred_category   = logclf_loaded.predict(clfv)

        url = eDoc.url

        sent_arry.append({"title": title, "content": content, "url": url, "category": pred_category, "sentiment": sentiment})

    return render_to_response("stocknews/index.html", { "news" : sent_arry })




