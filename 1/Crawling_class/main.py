import requests
import re
from bs4 import BeautifulSoup
import json
import pandas as pd
import numpy as np
reviews=[]
for i in range(1,300):
    try:
        Main_url = "https://www.theguardian.com/film+tone/reviews?page={0}".format(i)
        page = requests.get(Main_url)
        soup = BeautifulSoup(page.text,"html.parser")
        script_tag = soup.findAll('script', {'type': 'application/ld+json'})
        json_content = json.loads(script_tag[2].string)
        for j in range(0,len(json_content)):
            url =json_content['itemListElement'][j]['url']
            try:
                news = requests.get(url)
            except:
                df = pd.DataFrame(reviews)
                with open('data.json', 'w') as f:
                    json.dump(reviews, f, indent=2)
                df.to_csv("'reviews.csv', index=False")
                continue
            news_soup = BeautifulSoup(news.text, "html.parser")
            paragraphs= news_soup.findAll("p")
            temp =""
            stars = news_soup.find_all('div', class_="dcr-1tlk9ix")
            rating = sum(1 for star in stars if star.find('path').get('fill') == 'currentColor')
            for k in range(1,len(paragraphs)):
                temp+=(paragraphs[k].text)
            news_articles = [{'news_headline': news_soup.find("h1").text,
                              'news_firstline': paragraphs[0].text,
                              'rating': rating,
                              'news_article': temp
                              }
                             ]
            reviews.extend(news_articles)
        with open('data.json', 'w') as f:
            json.dump(reviews, f, indent=2)
    except:
        continue

df = pd.DataFrame(reviews)
df.to_csv('reviews.csv', index=False)
df.to_json("panda.json")
print(df.head(10))