import bs4 as bs4
import requests
import urlparse
import time
import sqlite3 as lite
from datetime import datetime as dt
import psycopg2
from dateutil.parser import parse

if __name__ == "__main__":

    t0 = dt.now()
    print "starting", t0

    url = 'http://finance.yahoo.com/news/provider-marketwatch/'
    url2 = "http://finance.yahoo.com/_xhr/top-story-temp/"

    form_data = "list_source=collection&apply_filter=0&filters=[]&content_id=6647ea16-dcee-3757-80a0-49b1b43f2f47&categories=%5B%5D&selected_tab=0&latest_on=0&s=1183300001&sec=MediaTopStoryTemp&story_start=51&storycount=50&popup_switch=1&provider_switch=1&timestamp_switch=1&max_title_length=100&more_inline=1&base_start=1&cache_ttl=TTL_LEVEL_10"

    # conn = lite.connect('data.sqlite') #/financenews
    # cur = conn.cursor()
    DBNAME = 'financenews'
    DBUSER = 'postgres'
    pf = open('password.txt')
    passwd = pf.readline()
    PASSWRD = passwd

    conn = psycopg2.connect(database= DBNAME, user=DBUSER, password=PASSWRD) #, host = '/tmp')
    cur = conn.cursor()

    s = requests.Session()
    s.head(url)

    # print s.headers

    sq = urlparse.parse_qsl(form_data)
    params = dict(sq)

    # with conn:
    #numArticles = int(raw_input('--Enter num of articles--'))
    numArticles = 201
    for i in range(0, numArticles, 50):
        #print 'article ',i
        params['story_start'] = i
        try:
            res = s.post(url2, data=params, headers={'Referer': url})
        except:
            time.sleep(10)
            res = s.post(url2, data=params, headers={'Referer': url})

        soup = bs4.BeautifulSoup(res.content)

        # print ">>>>>>>>>>>>>>>",soup.li
        try:
            soup.li.cite.text.encode('utf-8').split('-')[-1].strip()
        except:
            # print "PASSING"
            pass

        for article in soup.find_all('li'):
            #print "Title:"
            #print article.a.text.encode('utf-8')
            title = article.a.text.encode('utf-8').replace("'",'').replace('"','')
            url = article.a.get('href').encode('utf-8')
            try:
                date = parse( article.cite.text.encode('utf-8').split('-')[-1].strip() , tzinfos={'EST', -18000})
                #date = dt.strptime( article.cite.text.encode('utf-8').split('-')[-1].strip() , '%b %d %Y %I:%M%p')
            except:
                date = dt.now()
            content = ''
            print '\n', url, '\n', date
            try:
                more_articles = requests.get(url)
            except:
                # print 'in EXCEPT'
                more_articles = requests.get("http://finance.yahoo.com/"+url)

            if more_articles.status_code == 200:
                innerSoup = bs4.BeautifulSoup(more_articles.content) #.text)
                mainStory = innerSoup.find_all('article')#[0]
                # this pulls out 90% of articles with content
                if (len(mainStory) == 0):
                    mainStory = innerSoup.find_all("div", { "class" : 'articlePage'} )
                if (len(mainStory) == 0):
                    # print "trying body yom-art-content clearfix"
                    mainStory = innerSoup.find_all("div", { "class" : 'body yom-art-content clearfix'} )            
                #print "LENGTH ",len(mainStory)
                if (len(mainStory) > 0):
                    for eMainStory in mainStory:
                        innerP = eMainStory.findChildren('p', { "class" : "" })
                        for u in range(len(innerP)):
                            # print ' '.join(innerP[u].text.encode('utf-8').split())
                            # content += ' '.join(innerP[u].text.encode('utf-8').split())
                            content += ' '.join(innerP[u].text.encode('utf-8').replace('"','').replace("'",'').split())
                            # print content
                else: # this handles links that have videos/audio
                    innerP = innerSoup.find_all("div", { "class" : "currentVideoInfo"})
                    if len(innerSoup.find_all("div", { "class" : "currentVideoInfo"})) == 0:
                        # maybe it's a blog
                        innerP = innerSoup.find_all("div", { "class" : "liveBlog-header"})
                        # get all the blog comments
                        innerP2 = innerSoup.find_all('li')
                        [innerP.append(i) for i in innerP2]
                    if (len(innerP) > 0):
                        for i in range(len(innerP)):
                            eMainStory = innerP[i].findChildren('p')
                            for u in range(len(eMainStory)):
                                
                                # print ' '.join(eMainStory[u].text.encode('utf-8').split())
                                # content += ' '.join(eMainStory[u].text.encode('utf-8').split())
                                content += ' '.join(eMainStory[u].text.encode('utf-8').replace('"','').replace("'",'').split())
                                # print content
            #requests.close()
            # print content
            #print title
            # print url
            # print date

            with conn:
                # don't add duplicates
                check = "SELECT count(*) from data where title ='" + title + "'"
                #print check
                cur.execute(check)
                
                numArticles = cur.fetchone()[0] 
                #print numArticles
                if numArticles == 0:
                    print 'new article ',title
                    # whatis = 'INSERT INTO data (title, content, url, date) VALUES( "' + title + '","'+content+'","'+url+'","'+str(date)+'")'
                    whatis = "INSERT INTO data (title, content, url, date) VALUES( '" + title + "','"+content+"','"+url+"','"+str(date)+"')"
                    # print "HHHHHH ", whatis
                    # cur.execute("INSERT INTO data (title, content, url, date) VALUES(" + title + "," + content + " ," + url + ", "+ date + ")")
                    cur.execute(whatis)
                    # cur.execute("INSERT INTO data (title, content, url, date) VALUES('title' ,'content','url','date')")

            # print "\n"
        #time.sleep(0.5) 
    if conn:
        conn.close()

    print "elapsed time", dt.now() - t0
