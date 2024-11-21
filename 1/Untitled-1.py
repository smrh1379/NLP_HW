import json
 
# Opening JSON file
f = open('fars-news-1.json')
 
# returns JSON object as 
# a dictionary
data = json.load(f)
 
print(data[0]["title"])