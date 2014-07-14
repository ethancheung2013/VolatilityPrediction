from selenium import webdriver

driver = webdriver.Chrome()

numPages = 1
TOTAL = 2
while numPages < TOTAL:
	if numPages == 1:
		# url = 'http://www.boardcentral.com/#'
		url = 'http://finance.yahoo.com/news/provider-marketwatch/'
	else:
		url = driver.find_element_by_link_text("Next >>")

	driver.get(url)

	for i, element in enumerate(driver.find_elements_by_class_name('txt')):
	    if i == 0:  # for testing just print first element
	        print element.find_element_by_tag_name('a').text
	        iURL = element.find_element_by_tag_name('a').get_attribute('href')
	        driver.get(iURL)
	        # first look for 'article'
	        mainStory = driver.find_element_by_tag_name('article')
	        print mainStory.find_element_by_tag_name('p').text



