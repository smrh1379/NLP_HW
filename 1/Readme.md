# Movie Review Analysis Project

This project was an exciting dive into web crawling, data processing, and natural language processing (NLP), centered around movie reviews from [The Guardian](https://www.theguardian.com). The goal was to explore how we can extract meaningful insights from these reviews while showcasing different text processing techniques.

---

## What We Did

### Crawling the Data
We started by crawling data from The Guardian's movie review section. The focus was on collecting:
- Review headlines
- Ratings given to the movies
- The full review text
- A summary line that encapsulates the review

Once we had all this information, it was neatly stored in a JSON file to keep things organized and ready for analysis.

---

### Diving into the Data
The first task was to figure out the linguistic patterns in the reviews—specifically the adjectives and key phrases used. Here’s how we approached it:
1. **Preprocessing**: We cleaned the text by:
   - Removing punctuation
   - Lowercasing everything for consistency
   - Keeping only the useful parts of the text
2. **Challenge**: At first, we tried stemming (using Porter Stemmer), but it created some funny issues. For instance, it turned `strange` into `strang` and confused it as a verb instead of an adjective. Not helpful when you're trying to analyze movie reviews!
3. **Solution**: We skipped stemming and focused on clean tokenization instead. This worked much better for extracting the adjectives and key phrases.

The result? A list of the most commonly used descriptive words and phrases in the reviews.

---

### Extracting Movie Titles
The second task was to extract movie titles from the review headlines. This was a different challenge:
- We didn’t want to over-clean the text, as movie titles can sometimes include stopwords or special characters.
- Using tokenization and regex patterns, we identified and extracted movie names with pretty good accuracy.

By linking these titles back to the reviews, we enhanced the dataset, making it easier to analyze reviews for specific movies in the future.

---

## What We Learned
This project taught us a lot about:
- Building a complete workflow, from crawling to processing and analyzing text data.
- Overcoming challenges in text preprocessing, like finding the right balance between cleaning and preserving information.
- Using tools like SpaCy and regex effectively to extract meaningful insights.

---

This project wasn’t just about technical implementation—it was also a fun exploration of how language is used in movie reviews. From extracting adjectives to identifying movie titles, each step was a new challenge that helped deepen our understanding of NLP techniques.
