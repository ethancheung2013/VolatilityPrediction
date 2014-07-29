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

def getScrapedContent():

    DBNAME = 'newscontent'
    DBUSER = 'ethancheung'
    PASSWRD = open('password.txt').readline()

    conn = psycopg2.connect(database= DBNAME, user=DBUSER, password=PASSWRD)
    sql = 'SELECT * from stocknews_newscontent'
    df = psql.frame_query(sql, conn)

    return df

def combineHistVolColumn(contentDf, volDf):
    '''          
       NAME:
          combineHistVolColumn

       DESCRIPTION:
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
        NAME:  getHistoricalVolatility

        DESCRIPTION:
            Computes the daily historical volatility
            Returns: Dataframe of S&P using 1 day lag
    '''
    sp = web.DataReader('^GSPC', data_source='yahoo', start = '2012-11-15', end = '2014-12-31')
    sp['Log_Ret'] = np.log(sp['Close'] / sp['Close'].shift(1))
    sp['Volatility'] = pd.rolling_std(sp['Log_Ret'], window=2) * np.sqrt(252)
    
    return sp

if __name__ == '__main__':
    # get, merge, and clean dataframe
    sp_df      = getHistoricalVolatility()
    content_df = getScrapedContent()
    X, y       = combineHistVolColumn(content_df, sp_df)

    # vectorize text
    clf  = TfidfVectorizer(stop_words='english')
    clfv = clf.fit_transform(X)

    # cross validation
    X_train, X_test, y_train, y_test = train_test_split(clfv, y, test_size=0.2, random_state=42)

    # use naive bayes
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    ipdb.set_trace()
    # 1 estimator score method
    print "Estimator score method: ", clf.score(X_test, y_test)
    # 2 scoring parameter
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')

    print "Scoring parameter 'accuracy' from cross val: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

    # 3 scoring via metric functions
    # print average_precision_score(y_test, y_pred)
    print confusion_matrix(y_test, y_pred)


    
