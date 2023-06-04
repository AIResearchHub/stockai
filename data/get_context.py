from gnews import GNews
import snscrape.modules.twitter as sntwitter
import multiprocessing
import datetime
import pandas as pd
import os

from utils import preprocess


def __get_news(keyword, start, end, count):
    google_news = GNews(start_date=start, end_date=end, max_results=count)
    json_resp = google_news.get_news(keyword)


    date = []
    url = []
    text = []
    for j in json_resp:
        try:
            article = google_news.get_full_article(j["url"]).text
            date.append(datetime.datetime.strptime(
                j["published date"],
                "%a, %d %b %Y %H:%M:%S GMT"))
            url.append(j["url"])
            text.append(article)

        except Exception as e:
            pass

    df = pd.DataFrame(
        {
            "Date" : date,
            "Url"  : url,
            "Text" : text
        }
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df["Date"] = df.index
    return df


def _get_news(keyword, start, end, count):
    news = []

    temp_start = start
    temp_end = start + datetime.timedelta(days=1)

    while temp_end <= end:
        print(temp_start, " [news] ", keyword)
        new = __get_news(keyword=keyword,
                         count  =count,
                         start  =temp_start,
                         end    =temp_end)
        news.append(new)

        temp_start += datetime.timedelta(days=1)
        temp_end += datetime.timedelta(days=1)

    news = pd.concat(news)
    return news


def get_news(ticker, keyword, start=2010, end=2023, count=10):
    """
    count: Maximum number of articles to get and save per day
    """
    folderpath = f"context/news/{ticker}"
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    for i in range(end-start):
        path = folderpath + f"/{start+i}.json"
        if not os.path.exists(path):
            print(f"Creating {path}...")
            news = _get_news(keyword=keyword,
                             start  =datetime.datetime(start+i, 1, 1),
                             end    =datetime.datetime(start+i+1, 1, 1),
                             count  =count)
            news.to_json(path, orient="records", lines=True)
            print(f"{path} done.")

def __get_tweets(keyword, start, end, count):
    search = f"{keyword} since:{start} until:{end}"

    tweets = []
    num = 0
    generator = sntwitter.TwitterSearchScraper(search)

    for i, tweet in enumerate(generator.get_items()):
        if num >= count:
            break
        if tweet.lang == "en":
            tweets.append([tweet.date, tweet.content])
            num += 1

    tweets = pd.DataFrame(tweets, columns=["Date", "Text"])
    tweets = preprocess(tweets)
    tweets["Date"] = pd.to_datetime(tweets["Date"])
    tweets.set_index("Date", inplace=True)
    tweets["Date"] = tweets.index
    return tweets


def _get_tweets(keyword, start, end, count):
    """
    Loops through one year one day at a time
    """
    tweets = []
    temp_start = start
    temp_end = start + datetime.timedelta(days=1)

    while temp_end <= end:
        tweet = __get_tweets(keyword=keyword,
                             count  =count,
                             start  =temp_start.strftime('%Y-%m-%d'),
                             end    =temp_end.strftime('%Y-%m-%d'))
        tweets.append(tweet)

        temp_start += datetime.timedelta(days=1)
        temp_end += datetime.timedelta(days=1)

    tweets = pd.concat(tweets)
    return tweets


def get_tweets(ticker, keyword, start=2010, end=2023, count=50):
    """
    count: Maximum number of tweets to get and save per day
    """
    folderpath = f"context/tweets/{ticker}"
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    for i in range(end-start):
        path = folderpath + f"/{start+i}.json"
        if not os.path.exists(path):
            tweets = _get_tweets(keyword=keyword,
                                 start  =datetime.datetime(start+i, 1, 1),
                                 end    =datetime.datetime(start+i+1, 1, 1),
                                 count  =count)
            if len(tweets) != 0: tweets.to_json(path, orient='records', lines=True)


if __name__ == "__main__":
    from utils import read_sp500, read_ticker_to_name

    tickers = read_sp500("sp500.csv")
    names = read_ticker_to_name("sp500.csv")

    ps = []
    for i, ticker in enumerate(tickers[:50]):
        tweet_keyword = f"${ticker}"
        news_keyword = names[ticker]

        # p = multiprocessing.Process(target=get_tweets, args=(ticker, tweet_keyword))
        # p.start()
        # ps.append(p)
        p = multiprocessing.Process(target=get_news, args=(ticker, news_keyword))
        p.start()
        ps.append(p)

        if i % 3 == 0 and i != 0:
            for p in ps:
                p.join()
