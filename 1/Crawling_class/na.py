import requests
import re
from bs4 import BeautifulSoup
import json
for j in range(1,2):
    Main_url = "https://www.goal.com/en/news/{0}".format(j)
    page = requests.get(Main_url)
    soup = BeautifulSoup(page.text,"html.parser")
    script_tag = soup.findAll('script', {'type': 'application/ld+json'})

    for i in range(0,1):
        json_content = json.loads(script_tag[0].string)
        url = json_content['itemListElement'][i]['item']['url']
        news=requests.get(url)
        news_soup= BeautifulSoup(news.text,"html.parser")
        print(news_soup.prettify())