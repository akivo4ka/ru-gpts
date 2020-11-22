#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
import datetime


URL = 'https://psy-files.ru/solution/page/{}'
NUMBER_OF_PAGES = 10


def get_article_links():
    article_links = set()
    for page in range(1, NUMBER_OF_PAGES + 1):
        r = requests.get(URL.format(page))
        soup = BeautifulSoup(r.text)
        all_links = soup.find_all('a')
        for link in all_links:
            href = link.get('href')
            if href and 'solution' in href and 'page' not in href:
                article_links.add(href)
    return article_links


def clean_text(text):
    res = text.replace('\xa0', ' ').replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').strip()
    while res.find('  ') != -1:
        res = res.replace('  ', ' ')
    return res


def parse_and_save():
    article_links = get_article_links()
    train = []
    print("Parsing psy-files started")
    t1 = datetime.datetime.now()
    for url in article_links:
        r = requests.get(url)
        soup = BeautifulSoup(r.text)
        for div in soup.find_all('div', {'class': 'entry-content'}):
            text_article = clean_text(div.text)
            if len(text_article) < 1000:
                continue
            if text_article.find('Содержание') == 0:
                text_article = text_article[len('Содержание')+1:].strip()
            train.append('<s>' + text_article + '</s>')
    t2 = datetime.datetime.now()
    print("Parsing ended. Time: {}".format(t2-t1))
    print("Number of samples:", len(train))

    with open('psy_files.txt', 'wb') as f:
        train_text = '\n'.join(train)
        f.write(train_text.encode('UTF-8'))


if __name__ == "__main__":
    parse_and_save()
