import twitter
import pickle
import nltk
import sys
import re
import os

classifier = None

api = twitter.Api(consumer_key="",
                  consumer_secret="",
                  access_token_key="",
                  access_token_secret="")


def generate_search_results(query):
    """ Returns a list of text content in tweets for a given search query """

    return [result.full_text if result.full_text else result.text for result in api.GetSearch(query, count=200)]


def load_classifier():
  """ Unpickle and load classifier from standard path """
  
    global classifier

    with open('classifier.pickle', 'rb') as f:
        classifier = pickle.load(f)


def generate_featureset(text):
    """ Cleans text and returns a featureset """

    text = re.sub(r'\$\w*', '', text)

    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)

    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)

    # remove hashtags
    text = re.sub(r'#', '', text)

    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    text = text.strip().replace('\n', '')

    # Tokenize result
    token_list = nltk.word_tokenize(text)

    # Remove stopwords
    clean_token_list = [token for token in token_list if token not in nltk.corpus.stopwords.words("english")]

    # Lematize tokens
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lematized_tokens = [lemmatizer.lemmatize(token) for token in clean_token_list]

    # Extract Features
    features = dict([(token.lower(), True) for token in lematized_tokens])

    return features


def create_classifier(featx):
    """ Creates a NaiveBayes Classifier from the NLTK library, and trains it on sample tweets from the NLTK Corpus"""

     global classifier
    
    negfeats = [(featx(t), "neg") for t in nltk.corpus.twitter_samples.strings("negative_tweets.json")]
    posfeats = [(featx(t), "pos") for t in nltk.corpus.twitter_samples.strings("positive_tweets.json")]

    trainfeats = negfeats + posfeats

    classifier = nltk.classify.NaiveBayesClassifier.train(trainfeats)

    # Save classifier
    with open('classifier.pickle', 'wb') as f:
        pickle.dump(classifier, f)

        print("created classifier")


def analyze(search_results):
    """ Returns a descriptive string and the sentiment value as an integer """
    
    sentiment = classifier.classify_many([generate_featureset(tweet) for tweet in search_results])

    i = 0
    for s in sentiment:
        if s == "neg":
            i += 1
        else:
            i -= 1

    print("Tweets containing:", sys.argv[1],
          "have an average sentiment value of", i,
          "on a scale from", len(sentiment)*-1, "to", len(sentiment))


if __name__ == "__main__":

    arg = sys.argv[1]

    if "help" in arg or type(arg) != str:
        print("Usage guide:    python script.py 'European Union' ")
        quit()

    if os.path.isfile('classifier.pickle'):
        load_classifier()
        analyze(generate_search_results(arg))
    else:
        create_classifier(generate_featureset)
        analyze(generate_search_results(arg))
