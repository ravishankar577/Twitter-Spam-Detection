# Twitter Spam Filter Using Multinomial Naive Bayes Classification with TF-IDF tokenization and Snowball Text Processing

This projects main goal is to apply Multinomial Naive Bayes Classification to a labelled dataset after proccession of texts and tokenization. 

Main workflow of the project is as follows;

1. Loading the dataset
2. Preparing data for processing
3. Processing data(**Snowballing**)
  * Separate the sentence into individual words
  * Convert all letters to lowercase
  * Remove stopwords
  * Don't care for non-English words(depending on your dataset)
4. Vectorize each word to count occurences in the given sentence/data using [TF-IDF](http://www.tfidf.com/)
5. Visualise spam and ham data using word clouds with occurence weights.
6. Scramble dataset and separate data for training and testing purposes
7. Train the algorithm
8. Test the trained model with your test data to obtain accuracy and F1 scores using [Multinomial Naive Bayes](https://stats.stackexchange.com/questions/33185/difference-between-naive-bayes-multinomial-naive-bayes)
9. Save the the trained model (see **model.pkl**)
10. Twitter Authentication and tweet object catching functionalities 
11. Load the trained model
12. Feed the extracted tweet into the model and predict Spamminess/Hamminess

**Do not forget, Bayes Classification is based on probability. If you want more accurate results, use a bigger dataset. 

### To run the Project
python live_detection.py
```

