# Assignment 3 

## Summary

| Category             |  Score | Total Possible |
| -------------------- | :----: | :------------: |
| Overview             |    8   |       10       |
| WebCrawler           |   28   |       30       |
| Data Wrangling       |   18   |       20       |
| Machine Learning     |   25   |       30       |
| Reporting and Coding |    9   |       10       |
| **Total**            | **88** |     **100**    |

**Percent:** **88%**

---

## Overview

* Project motivation (finance-news link) is strong but presented informally at times.
* The link between sentiment modelling and financial impact is well outlined but lacks academic references.
* No clear research question — narrative is goal-driven but not inquiry-framed.

## WebCrawler

* Site map parsing shows strong scraping design and efficient structure.
* Scraping scripts are modular and respectful of `robots.txt`.
* Code demonstrates appropriate delays and parsing strategies.
* Crawled data is rich and well-documented; targets content, not metadata.

## Data Wrangling

* Sentiment scoring with both **VADER** and **FinBERT** is integrated smoothly.
* Data merging with financial time series is well explained.
* Handling of missing or sparse data is described but not deeply evaluated.
* Some preprocessing steps are assumed rather than justified (e.g. dropping neutral labels).
* Could improve with exploration of dataset imbalance or more rigorous statistical summary.

## Machine Learning

* Multiple models implemented: sentiment scoring, price prediction, RNN models.
* Metrics are discussed but not critically unpacked (e.g. performance vs. baseline).
* Good model breadth but lacks strong comparative argument for chosen architectures.

## Report & Coding

* Report is structured and walk through is complete, though informal in parts.
* Some figures are included without critical commentary.
* Academic voice is inconsistent — leans conversational in code-linked sections.
* Few references to academic literature beyond implementation-level documentation.
