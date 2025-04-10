import pandas as pd

is_article_url = lambda url: len(url) >= 50  # article titles are included in the url, so we can use this distinction to filter out non-article URLs
sitemap_count = 66
articles = []
for sitemap_number in range(1, sitemap_count + 1):
    print(f"Fetching sitemap {sitemap_number}")
    try:  # there are a list of sitemaps with the names part/1.xml, part/2.xml, we gather all urls from all sitemaps and append to articles list
        sitemap_sub_path = f'sitemap/part/{sitemap_number}.xml'
        article_metadata = extract_article_metadata_list_from_sitemap(f'https://smallcaps.com.au/{sitemap_sub_path}', sitemap_number)        
        article_metadata = list(filter(lambda url_data: is_article_url(url_data['url']), article_metadata))
        articles.extend(article_metadata)
    except Exception as e:
        print(f"Failed to fetch sitemap {sitemap_number}. Error: {e}")
df = pd.DataFrame(articles)
df.to_csv('../data/article_metadata.csv', index=False)