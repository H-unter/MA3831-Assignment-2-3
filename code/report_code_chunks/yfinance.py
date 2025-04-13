import yfinance as yf
import pandas as pd
import ast
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter


articles_df = pd.read_csv('../data/article_scraped_data.csv', converters={'stock_codes': ast.literal_eval}).dropna(subset=['document'])
articles_df = articles_df[articles_df['stock_codes'].apply(lambda x: len(x) > 0)]
articles_df.dropna(subset=['document'], inplace=True)
stock_codes_unformatted = articles_df['stock_codes'].explode().unique()
def change_stock_string_format(stock_code):
    if stock_code.startswith('ASX:'):
        return stock_code[4:] + '.AX'
    return stock_code
stock_codes = [change_stock_string_format(stock_code) for stock_code in stock_codes_unformatted]

start_date = articles_df['datetime'].min()
end_date = articles_df['datetime'].max()
# --- Download all tickers at once ---

class CachedLimiterSession(LimiterMixin, Session):
    pass
session = CachedLimiterSession(
    limiter=Limiter(RequestRate(2, Duration.SECOND * 5)),  # max 2 requests per 5 seconds
    bucket_class=MemoryQueueBucket,
    backend=SQLiteCache("yfinance.cache"),
)
df = yf.download(stock_codes, 
                 start=start_date, end=end_date, period='1d', 
                 group_by='ticker', 
                 session=session, 
                 threads=True, 
                 )

df.to_parquet('../data/stock_data.parquet', index=True, engine='pyarrow')