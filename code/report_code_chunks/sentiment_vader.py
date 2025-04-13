from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # apply lemminization first

analyzer = SentimentIntensityAnalyzer()
def get_vader_sentiment_scores(document, lemminize_document=False):
    """Clean the text, and calculate VADER sentiment scores"""
    if lemminize_document:
        document = [PorterStemmer().stem(word) for word in document.split()]
    return analyzer.polarity_scores(document)

sentiment_scores = articles_df['document'].apply(get_vader_sentiment_scores)

sentiment_scores_df = sentiment_scores.apply(pd.Series).rename(
    columns={'pos': 'vader_positive', 'neg': 'vader_negative', 'neu': 'vader_neutral', 'compound': 'vader_compound'}
)
articles_df = pd.concat([articles_df, sentiment_scores_df], axis=1)