import newspaper
papers =newspaper.build("https://www.goal.com/en/news",memoize_articles=False)
print(papers)