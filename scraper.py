import os
os.chdir('C:/Users/Nils/Desktop/proj')
import praw
r = praw.Reddit(user_agent='topic_model_scraper_by_rnilsmo')
# submissions = r.get_subreddit('BitcoinMarkets').get_new(limit=10000)

b1 = []
b2 = []
count = 0
for submission in r.get_new(limit=100000000):
    count += 1
    if count % 100 == 0:
        print submission.created_utc
    if submission.subreddit.display_name.lower() == 'bitcoin':
        b1.append(submission)
        
    if submission.subreddit.display_name.lower() == 'bitcoinmarkets':
        b2.append(submission)