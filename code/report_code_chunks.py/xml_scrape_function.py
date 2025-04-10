import requests
from bs4 import BeautifulSoup
from datetime import datetime

def extract_article_metadata_list_from_sitemap(url, sitemap_number):
    """
    Iterate through a sitemap and for each url tag, extracts the URL, title, last modified timestamp, year, month, and day.
    Returns a list of dictionaries, each containing the extracted metadata, for the given sitemap URL.
    """
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch the sitemap. Status code: {response.status_code}")
        return []
    
    article_observations = []
    soup = BeautifulSoup(response.content, 'lxml-xml')
    for url_tag in soup.find_all('url'):
        loc_tag = url_tag.find('loc')
        lastmod_tag = url_tag.find('lastmod')
        
        if loc_tag is None:
            continue # skip if <loc> is missing
            
        loc_tag_text = loc_tag.text.strip()
        last_modified = lastmod_tag.text.strip() if lastmod_tag else None

        # infer report title from url
        title = loc_tag_text.replace("https://smallcaps.com.au", "").replace("-", " ").replace("/", "").capitalize()
        
        lastmod_timestamp = None
        if last_modified:
            try: # convert last_modified to a datetime object
                lastmod_timestamp = datetime.strptime(last_modified, '%Y-%m-%dT%H:%M:%S%z').timestamp()
            except Exception as e:
                print(f"Error parsing date {last_modified}: {e}")

        article_observations.append({
            "url": loc_tag_text,
            "sitemap_number": sitemap_number,
            "title": title,
            "last_modified_timestamp": lastmod_timestamp, 
            "last_modified_year": last_modified[:4] if last_modified else None,  # Year format
            "last_modified_month": last_modified[5:7] if last_modified else None,  # Month format
            "last_modified_day": last_modified[8:10] if last_modified else None  # Day format
        })
    
    return article_observations