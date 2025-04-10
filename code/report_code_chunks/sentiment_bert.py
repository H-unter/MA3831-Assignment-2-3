from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

def get_bert_sentiment(doc):
  truncated_text = tokenizer.decode(tokenizer.encode(doc, max_length=510, truncation=True))
  result = nlp(truncated_text)[0]  # Call nlp only once
  return pd.Series({'bert_sentiment_label': result['label'], 'bert_sentiment_score': result['score']})

articles_df[['bert_sentiment_label', 'bert_sentiment_score']] = articles_df['document'].apply(get_bert_sentiment)
articles_df