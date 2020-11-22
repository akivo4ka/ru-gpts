#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
import datetime


URL = 'https://psy-practice.com/publications/?PAGEN_1={}'
NUMBER_OF_PAGES = 10


def get_article_links():
    article_links = set()
    for page in range(1, NUMBER_OF_PAGES + 1):
        r = requests.get(URL.format(page))
        soup = BeautifulSoup(r.text)
        all_links = soup.find_all('a')
        for link in all_links:
            href = link.get('href')
            href = link.get('href')
            if '/publications/' in href and '?' not in href and href[15:-1].find('/') != -1:
                article_links.add('https://psy-practice.com' + href)
    return article_links


def clean_text(text):
    res = text.replace('\xa0', ' ').replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').strip()
    while res.find('  ') != -1:
        res = res.replace('  ', ' ')
    return res


def parse_and_save():
    article_links = get_article_links()
    train = []
    print("Parsing psy-practice started")
    t1 = datetime.datetime.now()
    for url in article_links:
        r = requests.get(url)
        soup = BeautifulSoup(r.text)
        div = soup.find('div', {'class': 'newsContent textBlock'})
        article_text = clean_text(div.text)
        train.append('<s>' + article_text + '</s>')
    t2 = datetime.datetime.now()
    print("Parsing ended. Time: {}".format(t2-t1))
    print("Number of samples:", len(train))

    with open('psy_practice.txt', 'wb') as f:
        train_text = '\n'.join(train)
        f.write(train_text.encode('UTF-8'))


if __name__ == "__main__":
    parse_and_save()
