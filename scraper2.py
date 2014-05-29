from __future__ import print_function

from bs4 import BeautifulSoup
from urllib import urlopen
from datetime import datetime
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import LancasterStemmer
from nltk.corpus import wordnet, stopwords
import nltk
import string
import numpy as np
import matplotlib.pyplot as plt

MIN_PAGE = 1
MAX_PAGE = 486

import os
os.chdir('C:/Users/Nils/Desktop/proj')

heading_class = 'omc-blog-one-heading'
date_class = 'omc-date-time-one'

lemmer = WordNetLemmatizer()
stemmer = LancasterStemmer()

def init():
    global soups
    soups = [get_soup(i) for i in range(MIN_PAGE, MAX_PAGE)]


    
def driver():
    print('Getting soups')
    init()
    print('Getting headlines')
    global headlines
    headlines = get_headlines()
    print('Getting dates')
    global dates
    dates = get_dates()
    global volumes
    volumes = get_volumes()
    print('Stemming')
    global stembydate
    stembydate = stems_by_date(headlines, dates, filter=True, strip=True, stem=True, lower=True, min_count=10)
    global counts
    counts = word_counts(stembydate)

#
# scraping, etc
#

def get_price_file():
    return 'C:\Users\Nils\Desktop\proj\data\prices.txt'

def get_volumes():
    from dateutil.parser import parse
    ans = dict()
    for line in open(get_price_file()).readlines():
        if line.startswith('#'):
            continue
        # Date Open High Low Close Volume(BTC) Volume(USD) WeightedPrice
        entries = line.split('\t')
        date = parse(entries[0])
        volume = entries[5]
        ans[date] = float(volume)
    return ans

def get_changes():
    from dateutil.parser import parse
    ans = dict()
    last_close = None
    for line in open(get_price_file()).readlines():
        if line.startswith('#'):
            continue
        # Date Open High Low Close Volume(BTC) Volume(USD) WeightedPrice
        entries = line.split('\t')
        close = float(entries[4])
        if last_close is not None:
            date = parse(entries[0])
            change = close - last_close
            ans[date] = float(change)
        last_close=close
    return vol_series(ans)

def get_url(i):
    return 'http://www.bitcoin-domain.com/category/bitcoin-subreddit/page/' + str(i) + '/'

def get_mirror_url(i):
    if i==1:
        return 'C:\Users\Nils\Desktop\proj\pages\index.html'
    return 'C:\Users\Nils\Desktop\proj\pages\index.html.' + str(i-1)
    
def get_soup(i):
    # MIN_PAGE <= i <= MAX_PAGE
    url = get_mirror_url(i)
    page = open(url).read()
    # print i
    return BeautifulSoup(page)

def get_headlines():
    ans = []
    for soup in soups:
        for par in soup.find_all(class_=heading_class):
            headline = par.find('a').contents[0]
            headline = headline.encode('ascii', 'ignore')
            ans.append(headline)
    return ans
 
def get_dates():
    from dateutil.parser import parse
    ans = []
    for soup in soups:
        for i in soup.find_all(class_=date_class):
            date = str(i)[str(i).find('</b> ')+5:str(i).find(' |  ')]
            ans.append(parse(date))
    return ans

#
# stemming, etc
#

def stems_by_date(headlines, dates, min_count = None, max_count = None, lower=True, filter=True, strip=True, stem=True):
    # filter: whether to filter stopwords
    stembydate = dict()
    stops = stopwords.words('english')
    stops.append('bitcoin')
    stops.append('bitcoins')
    stops.append('btc')
    for i, headline in enumerate(headlines):
        if lower:
            headline = headline.lower()
        if strip:
            headline = headline.translate(None, string.punctuation)
        date = dates[i]
        cur = stembydate.setdefault(date, list())
        words = nltk.word_tokenize(headline)
        if filter:
            words = [w for w in words if w.lower() not in stops]
        if stem:
            words = [stemmer.stem(i) for i in words]
        cur = cur.extend(words)
    return filter_stembydate(stembydate, min_count, max_count)

def filter_stembydate(stembydate, min_count=None, max_count = None):
    if max_count is None:
        max_count = float('inf')
    if min_count or max_count:
        counts = word_counts(stembydate)
        for i, cur in stembydate.iteritems():
            cur2 = list()
            for word in cur:
                if counts[word] >= min_count and counts[word] <= max_count:
                    cur2.append(word)
            stembydate[i] = cur2
    return stembydate

def docs(stembydate):
    ans = []
    word_to_num = get_word_to_num(word_counts(stembydate))
    for date in stembydate:
        doc = []
        for word in stembydate[date]:
            doc.append(word_to_num[word])
        ans.append(doc)
    return ans

def vol_series(volumes):
    return [volumes[i] for i in sorted(volumes.keys())]

def word_counts(stembydate):
    ans = dict()
    for _, val in stembydate.iteritems():
        for stem in val:
            ans[stem] = ans.setdefault(stem, 0) + 1
    return ans

def get_word_to_num(counts):
    return dict(zip(sorted(counts.keys()), range(len(counts))))

def get_num_to_word(counts):
    return dict(zip(range(len(counts)), sorted(counts.keys())))

def most_common_words(counts):
    import operator
    return sorted(counts.iteritems(), key=operator.itemgetter(1), reverse=True)