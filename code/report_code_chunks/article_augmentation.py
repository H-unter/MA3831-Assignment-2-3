asx_companies_df = pd.read_csv('../data/ASXListedCompanies.csv')
asx_companies_df['code'] = "ASX:" + asx_companies_df['ASX code']
asx_companies_df

# article_df['stock_codes'] is a cell containing a list of stock code
# create a new column 'stock_code_instustries' which is a list of industries for each stock code found in the asx_comnpaies_df['GICS industry group']
def get_industries(stock_codes):
    if not stock_codes:  # Check if the list is empty
        return None
    industries = asx_companies_df.loc[asx_companies_df['code'].isin(stock_codes), 'GICS industry group']
    return set(industries) if not industries.empty else None

# Create a column for stock code industries (list of industries)
article_df_raw['stock_code_industries'] = article_df_raw['stock_codes'].apply(get_industries)

# Create a column for stock code industry (single industry)
def get_most_popular_industry(industries):
    if industries is None or not industries:  # Handle None or empty set
        return None
    industry_counts = pd.Series(list(industries)).value_counts()
    return industry_counts.idxmax()

article_df_raw['stock_code_industry'] = article_df_raw['stock_code_industries'].apply(get_most_popular_industry)
article_df_raw

# print out a list of all unique stock code industries
unique_industries = set()
for industries in article_df_raw['stock_code_industries']:
    if industries:  # Check if industries is not None
        unique_industries.update(industries)
unique_industries

industry_long_name_to_industry_short_name = {
    "Automobiles & Components": "Auto & Parts",
    "Banks": "Banks",
    "Capital Goods": "Capital Goods",
    "Class Pend": "Classification Pending",
    "Commercial & Professional Services": "Commercial Services",
    "Consumer Discretionary Distribution & Retail": "Retail",
    "Consumer Durables & Apparel": "Durables & Apparel",
    "Consumer Services": "Consumer Services",
    "Energy": "Energy",
    "Equity Real Estate Investment Trusts (REITs)": "Real Estate Invstmt.",
    "Financial Services": "Financial Services",
    "Food, Beverage & Tobacco": "F&B & Tobacco",
    "Health Care Equipment & Services": "Medical Equipment & Services",
    "Household & Personal Products": "Household & Personal Goods",
    "Materials": "Materials",
    "Media & Entertainment": "Media & Entertainment",
    "Not Applic": "Not Applicable",
    "Pharmaceuticals, Biotechnology & Life Sciences": "Pharma & Biotech",
    "Real Estate Management & Development": "Real Estate Mgt & Dev",
    "Semiconductors & Semiconductor Equipment": "Semiconductors",
    "Software & Services": "Software & Services",
    "Technology Hardware & Equipment": "Tech Hardware",
    "Telecommunication Services": "Telecom Services",
    "Transportation": "Transportation",
    "Utilities": "Utilities",
}
broad_sector_map = {
    "Auto & Parts": "Consumer Discretionary",
    "Banks": "Banking",
    "Capital Goods": "Industrial",
    "Classification Pending": "Other",
    "Commercial Services": "Industrial",
    "Retail": "Retail",
    "Durables & Apparel": "Consumer Discretionary",
    "Consumer Services": "Consumer Discretionary",
    "Energy": "Energy",
    "Real Estate Invstmt.": "Real Estate",
    "Financial Services": "Banking",
    "F&B & Tobacco": "Consumer Staples",
    "Medical Equipment & Services": "Healthcare",
    "Household & Personal Goods": "Consumer Staples",
    "Materials": "Mining",
    "Media & Entertainment": "Media",
    "Not Applicable": "Other",
    "Pharma & Biotech": "Healthcare",
    "Real Estate Mgt & Dev": "Real Estate",
    "Semiconductors": "Tech",
    "Software & Services": "Tech",
    "Tech Hardware": "Tech",
    "Telecom Services": "Telecom",
    "Transportation": "Industrial",
    "Utilities": "Utilities",
}

#map the long name to the short name in the dataframe
article_df_raw['stock_code_industry'] = article_df_raw['stock_code_industry'].map(industry_long_name_to_industry_short_name).map(broad_sector_map)
df = article_df_raw
df = df.dropna(subset=['document']).reset_index(drop=True)
df = df.loc[df['sector'] != 'Hot Topics']
df

# add a datetime column to the dataframe using pd.to_datetime
df['datetime'] = pd.to_datetime(df['date_string'], format='%B %d, %Y', errors='coerce')
df.to_csv('../data/article_scraped_data.csv', index=False)