from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd
import time
import re

driver = webdriver.Chrome('chromedriver')

res = []

for page in range(1, 10):
    driver.get('view-source:https://www.radiofrance.fr/franceinter/podcasts/le-7-9?p=' + str(page))
    time.sleep(1)
    html_source = driver.page_source

    pattern = r"https[\w/:.-]*?\.mp3"

    matches = re.findall(pattern, html_source)
    res += matches

    time.sleep(1)
    

pd.DataFrame({"URL": res}).to_csv('URLs.csv') 

driver.close()