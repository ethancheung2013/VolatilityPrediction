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
from sklearn.metrics import confusion_matrix, silhouette_score, roc_curve, auc
from sklearn.decomposition import NMF
import cPickle
import matplotlib.pyplot as plt
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords
from pandas.tseries.offsets import *
from dateutil import parser
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from time import time
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

sp_done = 0
def generate_tfidf(data):
    '''
        Description:
            Generates a tfidf vectorizer

        Parameters:
            Returns: a parameterized vectorizer
    '''
    n_features = 10000
    n_topics = 10
    n_top_words = 20
    
    # tfidf  = TfidfVectorizer(max_df=0.95, min_df=min_df, max_features=n_features, stop_words='english')
    tfidf  = TfidfVectorizer(max_features=n_features, stop_words='english')
    clfv = tfidf.fit_transform(data)

    # with open ('kmeans_tfidf.pkl', 'wb') as fid:
    #     cPickle.dump(tfidf, fid)

    return tfidf, clfv

def getConn(DBNAME, DBUSER, PASSWRD, tablename, bLabel=False):
    '''
        Description:
            Generic database connection

        Parameters:
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
        Description:
            Gets data from multiple datasources having URL, title, content, date and returns a merged dataframe

        Parameters:
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
       Description:
          Scraped content dataframe from postgres
          Volatility dataframe from google

       RETURNS:
          Merges the scraped web content with the historical volatility labels
          Precondition: Scraped content dataframe, volatility dataframe
          Returns: Content dataframe, label dataframe
    '''
    # reset the index to make the Date column available to be joined with other dataframe

    volDf = volDf.reset_index()

    volDf.columns = ['index', 'Volatility']
    #make date columns same type
    contentDf['Date_obj'] = pd.to_datetime(contentDf['date'])

    # ENHANCEMENT: calculate the volatility for each document per given time period
    volDf['Date_obj'] = pd.to_datetime(volDf['index'])

    merged = pd.merge(contentDf, volDf, on='Date_obj', how='outer')

    # handle NAs
    merged = merged.dropna(subset=['content', 'url'])
    X = merged[['content', 'url']]

    y = merged['Volatility']
    # fill in the weekends with 0 volatilty
    y = y.fillna(0)

    return X, y
 
def getHistoricalVolatility(time_period):
    '''
        Description:
            Builds the S&P daily historical volatility dataframe since 2012-11-15 to end of 2014

        Parameters:
            Returns: Dataframe of S&P using 1 day lag
    '''
    global sp_done

    if sp_done == 0:
        data_start = '2012-11-15'
        sp = web.DataReader('^GSPC', data_source='yahoo', start = data_start, end = '2014-12-31')

        sp['IntraDay_Vol'] = sp['High']-sp['Low']       # intra_vol will be proxy for volatility instead of taking std of closing prices
        sp.fillna(0, inplace=True)
        sp_done = 1
        sp.to_csv('hist_vol.csv')
    else:
        sp = pd.read_csv('hist_vol.csv', parse_dates=True, index_col='Date')

    # calculate the volatility delta over a defined range before and after each date
    vol_delta = pd.DataFrame()

    temp_dict = {}
    for eDate in sp.index.date:
        temp_dict[eDate] = volatility_delta(eDate, time_period, sp)

    vol_delta = vol_delta.from_dict(temp_dict, orient='index')
    vol_delta = vol_delta.sort_index()

    return vol_delta

def volatility_delta(doc_date, time_period, df):
    '''
        Description:
            Without exact time stamps for documents and minute financial data, assumes that documents for a given day contributes to the volatility for that day
            (i.e. documents are not separated by day)
        Parameters:
            Document date and the period of interest to calculate the volatility differences
            Data dataframe containing historical volatility
    '''
    before = df.ix[price_data(doc_date, time_period, df, True)]['IntraDay_Vol']
    before.fillna(0, inplace=True)
    after = df.ix[price_data(doc_date, time_period, df, False)]['IntraDay_Vol']
    after.fillna(0, inplace=True)

    v_before = volatility(before)
    v_after = volatility(after)
    vol_delta = v_before - v_after

    return vol_delta

def price_data(obj_doc_date, time_frame, data_df, bBefore):
    '''
        Description:
            Helper function for volatility_delta
            time_frame is measured in business days 
            doc_date is the median of the date range

            Returns the volatility data for the period before and after.  After period is inclusive of the document release date
            Receives data_df and indexed by .['2014-1-30': XXX]

            XXX is determined from the time_frame.  For example, 2014-1-23 if time_frame is week and 2014-2-6 after
    '''

    # make the start date
    if type(obj_doc_date) == 'str':
        obj_doc_date = parser.parse(obj_doc_date)
    # ensure that beginning period does not include the day the document is released
    if bBefore:
        data_window = pd.date_range(obj_doc_date - (time_frame + 1) * BDay(), obj_doc_date - 1 * Day())
    else:
        data_window = pd.date_range(obj_doc_date, obj_doc_date + time_frame * BDay())
    
    # get volatility for the period before and after
    return data_window #(data_df.ix[range_before]['intra_vol'], data_df.ix[range_after]['intra_vol'])

def volatility(price_data):
    '''
        Description:
            Helper function for volatility_delta
            Dataframe containing the price data of the S&P for a specified period

        Parameters:
            Receives data for a window period
            Returns volatility of price range data
    '''
    return np.std(price_data)


def displayScore(clf, X_train, y_train, X_test, y_test, y_pred):
    '''
        Description:
            Generalizable display score function

        Parameters:
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

def calculateSentiment(X):
    '''
        Description:
            Calculates sentiment as a feature of predicting volatility

        Parameters:
            Receives the corpus of documents
            Returns dataframe of one column containing sentiment
    '''
    nb_classifer = train_sentiment_classifier()
    tokenizer = RegexpTokenizer(r'\w+')

    sentDict = {}
    for idx, eDoc in enumerate(X):
        # classify the sentiment
        newdict = {}
        for i in tokenizer.tokenize(eDoc):
            newdict[i] = True
        sentDict[idx] = nb_classifer.classify(newdict)
    sentDf = pd.DataFrame.from_dict(sentDict, orient='index')
    sentDf = sentDf.sort_index()
    return sentDf

def linear_reg():

    # get content that is labeled using getScraped
    # Case 1: for supervised learning
    article_df                = getScrapedContent(True)
    df1_label                 = article_df['label']
    df1_content               = article_df[['content','date']]
    sp_df                     = getHistoricalVolatility()
    X, y_vol                  = combineHistVolColumn(df1_content, sp_df)
    X_sentiment               = calculateSentiment(X)
    # generate tfidf
    tfidf, clfv = generate_tfidf(X['content'])
    X_train, X_test, y_train, y_test = train_test_split(clfv, df1_label, test_size=0.4, random_state=42)

    clf = LinearRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    displayScore(clf, X_train, y_train, X_test, y_test, y_pred)

def word_feats(words):
    '''
        Description:
            Helper function for sentiment classifier

        Parameters:
            Receives words
            Returns dict
    '''

    return dict([(word, True) for word in words])

def train_sentiment_classifier():
    '''
        Description:
            Trains naive bayes classifier 
            Bootstrapped with basic movie word corpus

        Parameters:
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
    tfidf, clfv = generate_tfidf(X['content'])

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


def displayROC(X_test, y_test, clf, showAUC, i, volThres):
    '''
        Description:
            If showAUC = True, displays the ROC curve
        Parameters:
            Receives X_test, y_test and a classifier
            Returns a graph if showAUC = True else, returns the value of roc_auc
    '''

    probas_ = clf.predict_proba(X_test)

    fpr, tpr,thresholds = roc_curve(y_test, probas_[:,1]) 
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f " % roc_auc, i)

    if showAUC:
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC : ' + str(clf)[:10] + ' ' + str(i) + ',' + str(volThres))
        plt.legend(loc="lower right")
        plt.show()
    return roc_auc

def lin_regression(X, y):
    '''
        Description:
            Runs a multiple regression of the topics + polarity against the dependent variable volatility
            to determine the weights of each variable
        Parameters:
            Receives independent variables (topics + sentiment), dependent variable Volatility
            returns weights of each topic

    '''


    clf = LinearRegression()
    clf.fit(X,y)

def newTestClassify(text):
    '''
        Description:
            Used to classify new text

    '''
    textdata = ''.join([i if ord(i) < 128 else ' ' for i in text])

def add_Binary_Features(orig_Content, featureDf):
    '''
        Description:
            Calculates sentiment, source info, binarizes and adds to feature DataFrame
        Parameters:
            Receives original content dataframe, feature DataFrame
            Returns updated dataframe with binarized features
    '''
    sentDf = pd.DataFrame(calculateSentiment(orig_Content['content']))
    # binarize the sentiment and add one column

    dummies1 = pd.get_dummies(sentDf[0])
    featureDf['Sentiment'] = dummies1.iloc[:,:-1]       #[:,1:2]   second column is pos
    orig_Content['url'] = orig_Content['url'].apply(lambda x: x.split('.')[1])
    dummies2 = pd.get_dummies(orig_Content['url'])
    featureDf = pd.concat([featureDf, dummies2.iloc[:,:-1]], join='outer', axis=1)

    featureDf['Date'] = featureDf['Date'].apply(lambda x: pd.to_datetime(x).dayofweek)

    return featureDf

def importance_forest(data, label):
    """Compute feature importance using decision trees classifier.
  
      INPUT: data  -- numeric pandas dataframe with non-missing values
             label -- boolean pandas series with which to predict on
  
    OUTPUT: results sent to stdout
    """

    clf = ExtraTreesClassifier()
    clf.fit(data, label)
  
    for imp, col in sorted( zip(clf.feature_importances_, data.columns), key=lambda (imp, col): imp, reverse=True ):
        print "[{:.5f}] {}".format(imp, col)

def get_feature_matrix(df):
    '''
        Description:
            Takes dataframe containing:
                'content', 'data', 'url'
            Uses pickled NMF to transform tfidf to categories
            Adds binaries  'Sentiment', 'url', 'Date'
            Date is represented as day of the week feature

        Parameters:
            Receives dataframe containing 'content', 'date', 'url'
            Returns feature matrix

    '''

    with open ('/Users/ethancheung/Documents/zipfianacademy/FoxScraper/km_lrnmf_tfi_model.pkl', 'rb') as fid:
        n_clf, logclf_loaded, tfclf_loaded, nb_classifer = cPickle.load(fid)


def show_confusion_mat(y_test, y_pred, i, volThres):

    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print 'Precision :', precision, 'Recall :', recall

    print cm
    # Show confusion matrix
    plt.matshow(cm)
    plt.title('Confusion matrix ' + str(i) + ',' + str(volThres))
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return precision, recall

def nmf_logistic():

    # Case 2: for unsupervised/semisupervised
    article_df2               = getScrapedContent(False)
    df2_content               = article_df2[['content','date', 'url']]
    df2_date                  = article_df2['date']

    time_period = 250
    showAUC = False

    aRocScore = []
    r2 = []

    with open('my_csv.csv', 'a') as f:

        for iPer in xrange(time_period):

            rocauc = []
            toIterate = np.linspace(0.0, 4.0, num=5)
            # toIterate = [1]
            for volThres in toIterate:   

                sp_df = getHistoricalVolatility(iPer)
                X, y_vol = combineHistVolColumn(df2_content, sp_df)

                # when using nmf, X_test is used bc I want to discover latent topics
                n_topics = 10
                n_top_words = 15
                n_clf = NMF(n_components=n_topics, random_state=1)

                tfidf, clfv = generate_tfidf(X['content'])
                W = n_clf.fit_transform(clfv)
                H = n_clf.components_

                W_corr = pd.DataFrame(W)
                W_corr['Date'] = df2_date.values
                W_corr['Volatility'] = y_vol.values

                # add binarize features
                data = add_Binary_Features(X, W_corr)
                # this label is continuous and won't work for logistic/random forest
                label = data.pop('Volatility')

                ######################### VISUALIZATION #####################
                # feature_names = tfidf.get_feature_names()
                # for topic_idx, topic in enumerate(n_clf.components_):
                #     print("Topic #%d:" % topic_idx)
                #     print(" ".join( [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]] ))
                #     print

                #generalization of topics to categories FYI
                # topics = {4: "World news", 0: "Corporate", 2: "Housing", 9: "Consumer spending", 8: "Corporate Earnings", \
                #           7: "Energy", 6: "Employment", 5: "Government", 1: "Auto", 3: "General"}

                # importance_forest(data, label)

                # plt.bar(df2_date, label.values, width=0.7, edgecolor='none', color=(label>0).map({True: '#006600', False: 'r'}), 
                #     label="Vol Delta over Time",
                #     )
                # plt.legend(loc='best')
                # plt.ylabel(u'Volatility Delta')
                # plt.xlabel(u'Time')
                # plt.title('Volatility Delta over Time')
                # plt.grid()
                # plt.show()

                # ipdb.set_trace()
                # plt.scatter(x=str(df2_date), y=label, marker='o') #, label='Avg Volatility', c=label.values, alpha=0.6)
                # # (label.values > 0).map({True: 'r', False: 'b'})
                # plt.show()

                ################################        CLASSIFICATION
                                                ########  LOGISTIC #########
                t0 = time()
                lr_clf  = LogisticRegression(C=1, penalty='l1', tol=0.01)

                W_corr['HasVolatility'] = W_corr['Volatility'].apply(lambda x: 1 if x > volThres or x < -volThres else 0)

                label_ = W_corr.pop('HasVolatility')

                X_train, X_test, y_train, y_test = train_test_split(data, label_, test_size=0.4, random_state=42)  

                # scores = cross_val_score(lr_clf, X_train, y_train, cv=1)
                # print "%s -- %s" % (lr_clf.__class__, np.mean(scores))
                # print("done in %fs" % (time() - t0))

                lr_clf.fit(X_train, y_train)

                aucscore = displayROC(X_test, y_test, lr_clf, showAUC, iPer, volThres)
                y_pred = lr_clf.predict(X_test)
                if showAUC:
                    precision, recall = show_confusion_mat(y_test, y_pred, iPer, volThres)
                    rocauc.append((precision, recall))                

                # append to an array to show AUC over different time periods and thresholds
                # r2.append([iPer, volThres, aucscore])
                tmpdf = pd.DataFrame([[iPer, volThres, aucscore]])
                tmpdf.to_csv(f, header=False)

                                                ########  RANDOM FOREST #########

                # rf_clf = RandomForestClassifier(verbose=10, n_estimators=1, n_jobs=-1, max_features=None)

                # # scores = cross_val_score(rf_clf, X_train, y_train, cv=1)
                # # print "%s -- %s" % (rf_clf.__class__, np.mean(scores))
                # # print("done in %fs" % (time() - t0))

                # rf_clf.fit(X_train, y_train)
                # aucscore = displayROC(X_test, y_test, rf_clf, showAUC, iPer, volThres)
                # y_pred = rf_clf.predict(X_test)
                # if showAUC:
                #     precision, recall = show_confusion_mat(y_test, y_pred, iPer, volThres)



                ################################           PREDICTION
                                                ########  LINEAR REG #########

                # print "------------------- performing LINEAR REG"
                # X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.4, random_state=42)  
                # lin_clf = LinearRegression()
                # lin_clf.fit(X_train, y_train)

                # y_pred = lin_clf.predict(X_test)
                # displayScore(lin_clf, X_train, y_train, X_test, y_test, y_pred)

                # print ("Residual sum of squares: %.2f" %
                #         np.mean((lin_clf.predict(X_test) - y_test) ** 2))
                # # Explained variance score: 1 is perfect prediction
                # print ('Variance score: %.2f' % lin_clf.score(X_test, y_test))

                # pickle the clssifiers
                # nb_classifer = train_sentiment_classifier()

                # with open ('lr_lin_n_clf.pkl', 'wb') as fid:
                #     cPickle.dump((n_clf, lin_clf, lr_clf, rf_clf, tfidf, nb_classifer), fid)


        r2Df = pd.DataFrame(r2)
        r2Df.to_csv('r2Df.csv')
 


    print 
    # sBeta = calculate_CrossCorrelation(W_corr, n_topics)

    # cross_corr_Topics = np.argsort(sBeta)[::-1]

    # print 'cross_corr_Topics :',cross_corr_Topics


    #plt.hexbin(df['Day'], df['Threshold'], C=df['AUC'], bins=None, gridsize=5)


    print    

options = {     
    'km'  : kmeans_logistic,
    'nmf' : nmf_logistic,
    'lin' : linear_reg,
}

if __name__ == '__main__':
    cmd_option = sys.argv[1:]
    options[cmd_option[1]]()








    
