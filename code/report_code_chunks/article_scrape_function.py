import re
import urllib.request
from bs4 import BeautifulSoup

def extract_html_from_url(url):
    """
    given a url, return a soup object which contains the html content of the page
    """
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html_content = urllib.request.urlopen(request).read().decode('utf-8')
    soup = BeautifulSoup(html_content, 'lxml')
    return soup

def parse_html_to_dictionary(soup_obj):    
    """
    Parse the HTML content of a Small Caps article into a dictionary, extracting the title, author, date, sector, stock codes, and document.
    """
    
    title_tag = soup_obj.find("h1", class_="c-post-header__title")
    if title_tag: # Parse the title from the <h1> element; 
        title = title_tag.get_text(strip=True)
    else: # fall back to <title> if not found.
        title = soup_obj.title.get_text(strip=True) if soup_obj.title else None

    # parse the author from the header
    author_tag = soup_obj.find('a', class_='c-post-header__author-link')
    author = author_tag.get_text(strip=True) if author_tag else None

    # parse the date from the first <time> tag
    date_tag = soup_obj.find('time')
    date_string = date_tag.get_text(strip=True) if date_tag else None

    # parse the sector from the navigation
    sector = None
    nav_section = soup_obj.find("div", class_="c-post__navigation")
    if nav_section:
        sector_tag = nav_section.find("a", class_="c-post__navigation-link")
        sector = sector_tag.get_text(strip=True) if sector_tag else None

    # parse the stock codes
    stock_codes = []
    # look for any div whose class attribute contains "c-tags"
    tags_container = soup_obj.find( 
        lambda tag: tag.name == "div" and tag.get("class") and any("c-tags" in c for c in tag.get("class"))
    )
    if tags_container:
        stock_text_span = tags_container.find("span", class_="c-tags__text")
        if stock_text_span and "Stock Codes" in stock_text_span.get_text():
            # Try to find an inner container if available
            inner = tags_container.find("div", class_="c-tags__inner")
            if inner:
                stock_codes = [a.get_text(strip=True) for a in inner.find_all("a", class_="c-tags__item c-tags__item--link")]
            else:
                stock_codes = [a.get_text(strip=True) for a in tags_container.find_all("a", class_="c-tags__item c-tags__item--link")]

    # parse the article body
    body_tag = soup_obj.find("div", class_="c-rich-text c-post__rich-text")
    if body_tag:
        # Use "\n" as separator to preserve paragraph breaks
        document = body_tag.get_text(separator = " ", strip=True)
        document = re.sub(r'[^\w\s]', '', document) if document else None # use regex to remove any character that is not a word character or whitespace
    else:
        document = None

    return {
        'title': title,
        'author': author,
        'date_string': date_string,
        'sector': sector,
        'stock_codes': stock_codes,
        'document': document
    }