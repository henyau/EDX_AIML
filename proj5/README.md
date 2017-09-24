# **Movie sentiment analysis using Sci-Kit Learn** 
**A demonstration of using CountVectorizer, TfidfVectorizer and SGDClassifier**
### Henry Yau

---



The goals of this project are the following:
* Predict the sentiment of a movie review based on a small corpus of text using vectorized word count/frequency

This project uses the "Large Movie Review Dataset" from: 
http://ai.stanford.edu/~amaas/data/sentiment/

### Method description
1.) The collection of raw text files is first read and stripped of stop words and then inserted into a .csv file along with positive or negative sentiment score.

2.) Four methods are implemented and evaluated. A regularized linear model is created using SGDClassifier with a hinge loss and L1 penalty. The models use either counts of unigrams or bigrams computed using CountVectorizer or use term frequency-inverse document frequency (tf-idf) of unigrams or bigrams computed using TfidfVectorizer.

3.) The predictions are checked with cross_val_score.

### Results
    SGDUnigram()
		Accuracy: 0.84 (+/- 0.01)
    SGDUnigramTfidf()
		Accuracy: 0.74 (+/- 0.02)
    SGDBigram()
		Accuracy: 0.84 (+/- 0.02)
    SGDBigramTfidf() 
		Accuracy: 0.86 (+/- 0.01)	
	



