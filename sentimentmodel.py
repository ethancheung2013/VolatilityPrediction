# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd
import pandas.io.data as web
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, LinearRegression
import psycopg2
import pandas.io.sql as psql
import ipdb
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_score, roc_curve
from sklearn.decomposition import NMF
import cPickle
import matplotlib.pyplot as plt
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords

def generate_tfidf(data):
    '''
        DESCRIPTION:
            Generates a tfidf vectorizer

        PARAMETERS:
            Returns: a parameterized vectorizer
    '''
    # n_samples = 2000
    n_features = 10000
    n_topics = 10
    n_top_words = 20
    
    # ipdb.set_trace()
    # tfidf  = TfidfVectorizer(max_df=0.95, min_df=min_df, max_features=n_features, stop_words='english')
    tfidf  = TfidfVectorizer(max_features=n_features, stop_words='english')
    clfv = tfidf.fit_transform(data)

    # with open ('kmeans_tfidf.pkl', 'wb') as fid:
    #     cPickle.dump(tfidf, fid)

    return tfidf, clfv

def getConn(DBNAME, DBUSER, PASSWRD, tablename, bLabel=False):
    '''
        DESCRIPTION:
            Generic database connection

        PARAMETERS:
            Database name, user, password, tablename
            Returns: dataframe with URL, title, content, date
    '''
   
    conn = psycopg2.connect(database=DBNAME, user=DBUSER, password=PASSWRD)

    # flexible sql statement to return labeled and unlabeled data
    if not bLabel:
        sql = 'SELECT url, title, content, date, label from ' + tablename + ' where label is null'
    else:
        sql = 'SELECT  url, title, content, date, label from ' + tablename + ' where label is NOT null'

    df = psql.frame_query(sql, conn)
    conn.close()

    return df

def getScrapedContent(bLabel):
    '''
        DESCRIPTION:
            Gets data from multiple datasources having URL, title, content, date and returns a merged dataframe

        PARAMETERS:
            Returns a merged dataframe that hasn't been cleaned
    '''

    DBNAME = zip(['newscontent', 'financenews'],['stocknews_newscontent2', 'data2'])
    DBUSER = 'ethancheung'
    PASSWRD = open('password.txt').readline()

    rDf = pd.DataFrame()
    for eDB in DBNAME:
        rDf = rDf.append(getConn(eDB[0], DBUSER, PASSWRD, eDB[1], bLabel))
    return rDf

def combineHistVolColumn(contentDf, volDf):
    '''          
       DESCRIPTION:
          Scraped content dataframe from postgres
          Volatility dataframe from google

       RETURNS:
          Merges the scraped web content with the historical volatility labels
          Precondition: Scraped content dataframe, volatility dataframe
          Returns: Content dataframe, label dataframe
    '''
    # reset the index to make the Date column available to be joined with other dataframe
    volDf = volDf.reset_index()

    #make date columns same type
    contentDf['Date_obj'] = pd.to_datetime(contentDf['date'])
    volDf['Date_obj'] = pd.to_datetime(volDf['Date'])
    merged = pd.merge(contentDf, volDf, on='Date_obj', how='outer')

    # handle NAs
    merged = merged.dropna(subset=['content'])
    X = merged['content']

    y = merged['Volatility']
    y = y.fillna(0)

    return X, y
 
def getHistoricalVolatility():
    '''
        DESCRIPTION:
            Computes the daily historical volatility since 2012-11-15 to end of 2014

        PARAMETERS:
            Returns: Dataframe of S&P using 1 day lag
    '''
    sp = web.DataReader('^GSPC', data_source='yahoo', start = '2012-11-15', end = '2014-12-31')
    sp['Log_Ret'] = np.log(sp['Close'] / sp['Close'].shift(1))
    sp['Volatility'] = pd.rolling_std(sp['Log_Ret'], window=2) * np.sqrt(252)
    
    return sp

def displayScore(clf, X_train, y_train, X_test, y_test, y_pred):
    '''
        DESCRIPTION:
            Generalizable display score function

        PARAMETERS:
            Receives various test, train dataframes
            Returns: Nothing
    '''
    # 1 estimator score method                                                                     
    print "\nEstimator score method: ", str(clf.score(X_test, y_test)) + '\n'

    # 2 scoring parameter   
    try:                                                                       
        scores = cross_val_score(clf, X_train, y_train, cv=2, scoring='accuracy')
        print "Scoring parameter 'accuracy' from cross val: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        # 3 scoring via metric functions                                                               
        # print average_precision_score(y_test, y_pred)                                                
        print confusion_matrix(y_test, y_pred)
    except:
        # pass in the case of unevaluable sparse matrices
        pass


def linear_reg():

    # get content that is labeled using getScraped
    # Case 1: for supervised learning
    article_df                = getScrapedContent(True)
    df1_label                 = article_df['label']
    df1_content               = article_df[['content','date']]
    sp_df                     = getHistoricalVolatility()
    X, y_vol                  = combineHistVolColumn(df1_content, sp_df)
    # generate tfidf
    tfidf, clfv = generate_tfidf(X)
    X_train, X_test, y_train, y_test = train_test_split(clfv, df1_label, test_size=0.4, random_state=42)

    clf = LinearRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    displayScore(clf, X_train, y_train, X_test, y_test, y_pred)

def word_feats(words):
    '''
        DESCRIPTION:
            Helper function for sentiment classifier

        PARAMETERS:
            Receives words
            Returns dict
    '''

    return dict([(word, True) for word in words])

def train_sentiment_classifier():
    '''
        DESCRIPTION:
            Trains naive bayes classifier

        PARAMETERS:
            Returns: a NB trained classifier
    '''
    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')

    negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
 
    negcutoff = len(negfeats)*3/4
    poscutoff = len(posfeats)*3/4

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
 
    classifier = NaiveBayesClassifier.train(trainfeats)
    return classifier

def kmeans_logistic():

    # Case 2: for unsupervised/semisupervised
    article_df2               = getScrapedContent(False)
    df2_content               = article_df2[['content','date']]
    df2_date                  = article_df2['date']
    sp_df                     = getHistoricalVolatility()
    X, y_vol                  = combineHistVolColumn(df2_content, sp_df)

    # generate vectorized clfv
    tfidf, clfv = generate_tfidf(X)

    clf = KMeans(n_clusters=10, init='k-means++', max_iter=100) #, n_init=1)
    clf.fit_predict(clfv)
    labels = clf.labels_

    # with open ('kmeans_km_model.pkl', 'wb') as fid:
    #     cPickle.dump(clf, fid)

    print '\nSilouette score :', str(silhouette_score(clfv, labels, metric='euclidean')) + '\n'

    X_train, X_test, y_train, y_test = train_test_split(clfv, labels, test_size=0.4, random_state=42)        

    clf_lr = LogisticRegression()
    clf_lr.fit(X_train, y_train)

    y_pred = clf_lr.predict(X_test)

    with open ('km_lr_tfi_model.pkl', 'wb') as fid:
        cPickle.dump((clf_lr, tfidf), fid)
    
    displayScore(clf_lr, X_train, y_train, X_test, y_test, y_pred)

def calculate_CrossCorrelation(df, n_topics):
    '''
        DESCRIPTION:
            Calculates Beta (i.e. tendency) of a given 1-D series with Volatility
            Assumes incoming dataframe has Volatility column

        PARAMETERS:
            Receives: Dataframe
            Returns: Numpy array
    '''
    sBeta = []
    for nTopics in xrange(n_topics):
        cov = np.cov(df[nTopics], df['Volatility'])
        ipdb.set_trace()
        sBeta.append(cov[1, 0] / cov[0, 0])

    return sBeta

def displayROC(X_test, y_test, clf):

    probas_ = clf.predict_proba(X_test)

    ipdb.set_trace()

    fpr, tpr,thresholds = roc_curve(y_test, probas_) 
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Region operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def nmf_logistic():

    # Case 2: for unsupervised/semisupervised
    article_df2               = getScrapedContent(False)
    df2_content               = article_df2[['content','date']]
    df2_date                  = article_df2['date']
    sp_df                     = getHistoricalVolatility()
    X, y_vol                  = combineHistVolColumn(df2_content, sp_df)

    # when using nmf, X_test is used bc I want to discover latent topics
    n_topics = 10
    n_top_words = 15
    n_clf = NMF(n_components=n_topics, random_state=1)
    # this is document to topic matrix - shows percentage of each topic to each article
    # i will try to show the correlation of each topic to the volatility
    #  doc/day  |   retail sales   |    employmt   |    VOLATILIITY
    #   1             .238                .145           .05
    #   2             .123                .1             .02
    # pull in date in column and take a moving average window of one week
    tfidf, clfv = generate_tfidf(X)
    W = n_clf.fit_transform(clfv)
    H = n_clf.components_

    W_corr = pd.DataFrame(W)

    W_corr['Date'] = df2_date.values
    W_corr['Volatility'] = y_vol.values

    sBeta = calculate_CrossCorrelation(W_corr, n_topics)

    cross_corr_Topics = np.argsort(sBeta)[::-1]

    print 'cross_corr_Topics :',cross_corr_Topics

    feature_names = tfidf.get_feature_names()
    for topic_idx, topic in enumerate(n_clf.components_[cross_corr_Topics]):
        print("Topic #%d:" % cross_corr_Topics[topic_idx])
        print(" ".join( [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]] ))
    print
    # give topic name
    # pass topic name to LR
    topics = {4: "World news", 0: "Corporate", 2: "Housing", 9: "Consumer spending", 8: "Corporate Earnings", \
              7: "Energy", 6: "Employment", 5: "Government", 1: "Auto", 3: "General"}
    # build label vector for each document
    labels = []
    for iRow in xrange(W_corr.shape[0]):
        category = topics[np.argmax(W_corr.iloc[iRow,:10])]
        labels.append(category)
    ipdb.set_trace()
    X_train, X_test, y_train, y_test = train_test_split(clfv, labels, test_size=0.4, random_state=42)        

    clf_lr_nmf = LogisticRegression()
    clf_lr_nmf.fit(X_train, y_train)

    y_pred = clf_lr_nmf.predict(X_test)

    ipdb.set_trace()
    #create and train nb classifier for the pickle
    nb_classifer = train_sentiment_classifier()

    with open ('km_lrnmf_tfi_model.pkl', 'wb') as fid:
        cPickle.dump((clf_lr_nmf, tfidf, nb_classifer), fid)
    
    displayScore(clf_lr_nmf, X_train, y_train, X_test, y_test, y_pred)
    
    #displayROC(X_test, y_test, clf_lr_nmf)

    print    

options = {     
    'km'  : kmeans_logistic,
    'nmf' : nmf_logistic,
    'lin' : linear_reg,
}

if __name__ == '__main__':
    cmd_option = sys.argv[1:]
    options[cmd_option[1]]()








    
