FoxScraper
==========
foxnews.py is a python scraper for http://www.foxnews.com/us/economy/index.html.  It runs a sellenium web driver that automates web scraping for more complex web pages.  It was used to automate the click of a 'Show More' button that reloads more economic news

scraper.py is a python scraper for http://finance.yahoo.com/news/provider-marketwatch/.  

yahoonews.py is a python scraper that pulls all publicly released 2014 economic release data on their Calendar page

initialclaims.py is NOT complete.  It is intended to scrape the DOL website and pull all Initial Claims and Continuing Claims economic releases using the selenium web driver.

alter.sql is used to drop the "NOT NULL" constraint on the postgres schema generated using Django

sentimentmodel.py contains all the models that will try to predict a stock volatility event
	'km'  : kmeans_logistic,
    'nmf' : nmf_logistic,
    'lin' : linear_reg,

    The NMF, Logistic Regression, Random Forest model is run using the command:  python sentimentmodel.py -m nmf
    There are additional functions to display the performance metrics of these models but will need to be uncommented.
    As well, the model is set to run for the last 250 days.  i.e will calculate the last 250 days of delta volatility with an additional inner loop to vary the delta threshold
