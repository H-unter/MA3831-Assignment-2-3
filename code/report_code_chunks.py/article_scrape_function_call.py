import ast
import pandas as pd

# Initialize an empty DataFrame to store the articles
crawl_throttle_time_s = 3 

# article_metadata was attained from the sitemaps' xml files earlier
article_metadata = pd.read_csv('../data/article_metadata.csv')
# Defining the existing data allows for scraping in installments, rather than in one go
# we therefore read the existing data (assuming we have already scraped some articles)
article_df_raw = pd.read_csv('../data/article_scraped_data_raw.csv', 
                         converters={'stock_codes': ast.literal_eval}
                         ) # raw data indicattes it has not been augmented yet
unscraped_articles = article_metadata.loc[article_metadata['url'].isin(article_df_raw['url']) == False].reset_index(drop=True)
unscraped_articles