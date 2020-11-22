#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
import datetime

URL = r'https://www.b17.ru/article/?page={}&sort=2'
NUMBER_OF_PAGES = 50


def get_article_links():
    article_links = []
    for page in range(1, NUMBER_OF_PAGES + 1):
        r = requests.get(URL.format(page))
        soup = BeautifulSoup(r.text)
        all_links = soup.find_all('a')
        for link in all_links:
            href = link.get('href')
            if href and 'article' in href and '?' not in href and href != '/article/':
                article_links.append('https://www.b17.ru' + href)
    return article_links


def clean_text(text):
    res = text.replace('\xa0', ' ').replace('\n', ' ').strip()
    return res


def parse_and_save():
    article_links = get_article_links()
    train = []
    print("Parsing b17 started")
    t1 = datetime.datetime.now()
    for url in article_links:
        r = requests.get(url)
        soup = BeautifulSoup(r.text)
        if not soup:
            continue
        article_div = soup.find('div', {'itmprp': 'articleBody'})
        if not article_div:
            continue
        list_article = article_div.find_all('p')
        text_article = []
        for item in list_article:
            item_text = clean_text(item.text)
            if item_text != '':
                text_article.append(item_text)
        text_article = '\n'.join(text_article)
        if len(text_article) < 1000:
            continue
        train.append('<s>' + text_article + '</s>')
    t2 = datetime.datetime.now()
    print("Parsing ended. Time: {}".format(t2-t1))
    print("Number of samples:", len(train))

    with open('b17.txt', 'wb') as f:
        train_text = '\n'.join(train)
        f.write(train_text.encode('UTF-8'))


if __name__ == "__main__":
    parse_and_save()
