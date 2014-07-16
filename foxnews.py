from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import time
import numpy as np
import psycopg2
from dateutil import parser
import ipdb
from datetime import datetime as dt

def getHref(webelements):
#        for i, element in enumerate(webelements):
#            print element.find_element_by_tag_name('a').get_attribute('href')
    return [wElement.find_element_by_tag_name('a').get_attribute('href') for wElement in webelements]

def showmore():
    # get the next page
    for i, element in enumerate(browser.find_elements_by_class_name("more-btn")):
        print element.find_element_by_tag_name('a').get_attribute('href')

def connect_to_db():
    ''' 
    NAME
        connect_to_db

    SYNOPSIS
        Connect to postgres

    DESCRIPTION
        Connect to postgres at __main__

    '''  

    # query database for a list of urls and put in set

    DBNAME = 'newscontent'
    DBUSER = 'ethancheung'

    pf = open('password.txt')
    passwd = pf.readline()
    PASSWRD = passwd
    #ipdb.set_trace()
    return psycopg2.connect(database= DBNAME, user=DBUSER, password=PASSWRD) #, host = '/tmp')       

def getContent(setOfURL):

    #check for duplicates        
    conn = connect_to_db()                                          
    cur = conn.cursor()
    with conn:
        allURLs = "SELECT url from stocknews_newscontent" 

        cur.execute(allURLs)
        existingURL = set(eItem[0] for eItem in cur.fetchall())
        
    newURL = setOfURL.difference(existingURL)

    for oneURL in newURL:
        browser.get(oneURL)
        strTitle = ''
        datePub = ''
        articleHeader = browser.find_elements_by_xpath('//*[@id="content"]/div/div[1]/div[2]/div/div[3]/article')
        if len(articleHeader) > 0:
            for eHeader in articleHeader[0].find_elements_by_tag_name('h1'):
                strTitle = eHeader.text

            for dHeader in articleHeader[0].find_elements_by_tag_name('time'):
                datePub = dHeader.get_attribute('datetime')
        # retrieve the content via p tags    
        mainStory = browser.find_elements_by_xpath('//*[@id="content"]/div/div[1]/div[2]/div/div[3]/article/div/div[3]')

        if len(mainStory) == 1:            
            content = ''
            for eContent in mainStory[0].find_elements_by_tag_name('p'):
                content += ''.join(eContent.text)

        storeContent(strTitle, datePub, content, oneURL)
    return True

def storeContent(strTitle, datePub, content, iurl):
    '''
    NAME
            storeContent

    SYNOPSIS
            Stores raw string data to Postgres using 'psql newscontent' generated with Django
            
    DESCRIPTION
           Stored variables:

                id      | integer                  | not null default nextval('stocknews_newscontent_id_seq'::regclass)
                url     | character varying(1000) 
                title   | character varying(4000) 
                content | character varying(50000)
                date    | date                    

           table: stocknews_newscontent       
    '''
    strContent = content.replace("'","")
    conn = connect_to_db() 
    cur = conn.cursor()


    dateObj = str(parser.parse(datePub.strip() , tzinfos={'EST', -18000}))
    if dateObj == '' or dateObj is None:
        dateObj = dt.now()

    strTitle = strTitle.replace("'","")
    sql = "INSERT into stocknews_newscontent (url, title, content, date) values ('"+ iurl + "','" + strTitle + "','" + strContent + "','" + datePub + "');"
    try: 
        cur.execute(sql)
    except:
        datePub = dt.now()
        cur.execute(sql)

    conn.commit()
    return "The item : %s was successfully saved to the databse" % strTitle

def main():
    '''
    NAME: main

    SYNOPSIS:
       Macro economic news scraper for Fox News

    DESCRIPTION:
       'Show More' button displays an extra 10 links
       Scraper clicks the 'Show More' button a configured number of times
       Then opens each link individual and scrapes the content going through H3 tags
    '''
    url = 'http://www.foxnews.com/us/economy/index.html#'

    browser.get(url)
    numClicks = 1
    advanceClicks = 400
    while numClicks < advanceClicks:
	if numClicks == 1:
            #url = 'http://www.foxnews.com/us/economy/index.html#'
            for eClick in np.arange(advanceClicks):
                # find the Show More button and click it a bunch of times
                browser.find_element_by_class_name("btn-smll").click()
                time.sleep(1)
                numClicks += 1

    # all news headlines are li but on under ul
    setURL = set()
    for i, element in enumerate(browser.find_elements_by_xpath('//*[@id="section-content"]/div[1]/div[4]/div/div/div[6]/div/div/div/ul/li')):
        try:
            h3Element = element.find_element_by_tag_name('h3')
#            ipdb.set_trace()
            setURL.add(h3Element.find_element_by_tag_name('a').get_attribute('href'))
        except: # NoSuchElementException:
            pass

    #strTitle, datePub, strContent, strUrl  = getContent(setURL)

    getContent(setURL)

browser = webdriver.Safari()   #Chrome()    
if __name__ == '__main__':
    main()

