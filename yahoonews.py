from datetime import datetime as dt
import numpy as np
import bs4 as Soup
import requests
import pandas as pd
import re
from sqlalchemy import create_engine

def cleanText(text):
    '''                                                                                                                                                              
    NAME: cleanText

    SYNOPSIS:                                                                                                                                                        
         To remove non-numeric characters that will interfere with determining whether an economic report is positive/negative

    DESCRIPTION:                                                                                                                                                         
        Precondition: Receives a string of text
        Postcondition: Removes dashes, NA explicitly.  Additionally, will remove any other characters like $ separately
    '''

    if text == '-' or text == '--':
        return ''
    cleaned = ''
    dash = False
    if '-' in text:
        dash = True
    try:
        cleaned = re.search("[0-9.]+",text).group(0)
    except:
        cleaned = re.search("[0-9.]*",text).group(0)
    if dash:
        cleaned = "-" + cleaned

    return cleaned

def createCalendarLinks():
    '''                                                                                                                                                              
    NAME: createCalendarLinks                                                                                                                                           

    SYNOPSIS:                                                                                                                                                        
          Yahoo economic calendar pages are filed by weeks and stored in a list object histEconReports
    DESCRIPTION:                                                                                                                                                                     A list object that stores the url links to Yahoo economic calendar pages with expectations, forecasts since beginning of 2014 
          The data is used to label official macro economic reports
          Precondition: None
          Postcondition: List object histEconReports
    '''
    now = dt.now()
    Year,WeekNum,DOW = now.isocalendar()
    histEconReports = []
    for eWeek in np.arange(1,WeekNum+1):
        if eWeek < 10:
            eWeek = '0' + str(eWeek)
        histEconReports.append('http://biz.yahoo.com/c/ec/2014' + str(eWeek) + '.html') 
    return histEconReports

def saveViaSQLalchemy(df):
    ''''
    NAME: saveViaSQLalchemy

    SYNOPSIS:                                                                                                                                                                     Saves 

    DESCRIPTION:                                               
        Precondition: Old dataframe of column types strings
        Postcondition: First three columns are strings, remainder are floats
    '''
    engine = create_engine('postgresql://ethancheung@localhost:5432/newscontent')
    engine.connect()
    table_name = 'stocknews_yahoocalendar'
    df.to_sql(table_name, engine)
    return True

def fixColumnTypes(df):
    ''''
    NAME: fixColumnTypes

    SYNOPSIS:                                                                                                                                                                     Converts columns 3 through 9 (non-inclusive) to float

    DESCRIPTION:                                               
        Precondition: Old dataframe of column types strings
        Postcondition: First three columns are strings, remainder are floats
    '''
    tofloat = [ u'ForActual', u'Briefing_Forecast', u'Market_Expects', u'Prior', u'Revised_From']
    df[tofloat] = df[tofloat].astype(float)
            
    return df

def retrieveEconData():
    ''''
    NAME: retrieveEconData                                                                                                                                                                           
    SYNOPSIS:                                                                                                                                                                       Entry function to retrieve Yahoo Finance economic expectations, actuals from Economic Calendar for 2014 till current

    DESCRIPTION:                                                                                                                                                        
    '''
    histEconReports = createCalendarLinks()

    yahoo_data = []
    for eHistEconReports in histEconReports:
        res = requests.get(eHistEconReports)
        print
        print eHistEconReports
        soup = Soup.BeautifulSoup(res.content)
        table = soup.findAll('table')
        trRows = table[5].findAll('tr')

        for i, eRow in enumerate(trRows):
            listDF = []
            if i == 0:  # skip the header row
                continue
            cols = eRow.findAll('td')
            for eString in np.arange(0,3):
                listDF.append(cols[eString].text)
            # for columns 3 - 8 clean data
            for sIdx in np.arange(4,9):
                listDF.append(cleanText(cols[sIdx].string))
            yahoo_data.append(listDF)

    df = pd.DataFrame(yahoo_data, columns=['Date','Time_ET','Statistic','ForActual','Briefing_Forecast','Market_Expects','Prior','Revised_From'])
    df = df.fillna(0)
    df = df.where(df != '',0)

    # reconstruct df to proper data types
    df = fixColumnTypes(df)
    # label each record as 1 for positive if Actual is > avg of Forecast + Expects
    df['label'] = np.where(df['ForActual'] > (df['Briefing_Forecast']+df['Market_Expects'])/2, 1, 0)

    # send to to sqlalchemy
    rVal = saveViaSQLalchemy(df)
    if rVal:
        print "Economic calendar successfully recorded into the database"
    else:
        print "There was a problem saving to the database"


if __name__ == '__main__':
    retrieveEconData()
