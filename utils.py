from google.cloud import storage
import pandas as pd
import random
import json
import os


def read_sp500(filename="data/sp500.csv"):
    """Returns a list of S&P500 tickers read from a .csv file"""
    file = pd.read_csv(filename)
    tickers = file["Symbol"].tolist()
    return tickers


def read_tickers():
    """Return usable tickers that have completed news article datasets (no gaps between years)"""
    path = f"data/context/news/"
    tickers = os.listdir(path)
    tickers = filter_tickers(tickers)
    return tickers


def read_bert_config(model_type):
    """Read bert configuration from pretrained_model/"""
    path = f"pretrained_model/{model_type}_config.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data


def filter_tickers(tickers):
    """if context news data incomplete then filter out ticker (bad data)"""
    filtered = []

    for ticker in tickers:
        path = f"data/context/news/{ticker}"
        if os.path.exists(path):
            years = os.listdir(path)

            if len(years) < 13:
                faulty = True
            else:
                faulty = False
                for year in years:
                    data = pd.read_json(f"{path}/{year}", lines=True)
                    if len(data) == 0:
                        faulty = True
                    if "Ids" not in data.columns:
                        faulty = True
        else:
            faulty = True

        if not faulty:
            filtered.append(ticker)

    return filtered


def read_ticker_to_name(filename="data/sp500.csv"):
    """Return a dictionary that maps tickers to names of the companies"""
    file = pd.read_csv(filename)
    tickers = file["Symbol"].tolist()
    names = file["Description"].tolist()

    res = {tickers[i]: names[i].replace(" Inc", "").replace(" Corp", "")
           for i in range(len(tickers))}
    return res


def read_prices(tickers, start, end, repeat=1):
    """Read prices from .csv files in /data/prices/ directory"""
    prices = []
    for ticker in tickers:
        price = pd.read_csv(f"data/prices/{ticker}.csv")
        price["Date"] = pd.to_datetime(price["Date"])
        price.set_index("Date", inplace=True)

        prices.append(price)

    prices = pd.concat(prices, axis=1, keys=tickers)
    prices = prices.iloc[:, prices.columns.get_level_values(1) == "Close"]
    prices = prices.loc[start:end]

    prices = prices.loc[prices.index.repeat(repeat)]

    return prices


def read_context_file(filename):
    """Helper function for read_context to read individual files"""
    try:
        data = pd.read_json(filename, lines=True)
        data["Date"] = pd.to_datetime(data["Date"])
        data.set_index("Date", inplace=True)

    except Exception as e:
        print(f"ERROR: Filename {filename}")
        print(e)
        print(pd.read_json(filename, lines=True))

    return data


def read_context(tickers, mock_data):
    """
    This function retrieves the news articles from directory
    /news/{ticker}
    /tweets/{ticker}

    Parameters:
    tickers (List[String]): List of tickers to be read from file
    mock_data (bool): If true, only one segment is returned for each date for each ticker

    Returns:
    context (dict[String, pd.Dataframe]): A dictionary with tickers as keys and dataframe of texts as value
    """
    context = dict()
    srcdir = "data/context"
    for ticker in tickers:
        news = []
        news_path = f"{srcdir}/news/{ticker}"
        if os.path.exists(news_path):
            years = os.listdir(news_path)
            for year in years:
                news.append(read_context_file(f"{news_path}/{year}"))
        if len(news) != 0: news = pd.concat(news)
        else: news = pd.DataFrame(columns=["Date", "Url", "Text", "Ids"])

        tweets = []
        tweets_path = f"{srcdir}/tweets/{ticker}"
        if os.path.exists(tweets_path):
            years = os.listdir(tweets_path)
            for year in years:
                tweets.append(read_context_file(f"{tweets_path}/{year}"))
        if len(tweets) != 0: tweets = pd.concat(tweets)
        else: tweets = pd.DataFrame(columns=["Date", "Url", "Text", "Ids"])

        # final processing
        c = pd.concat([news, tweets]).sort_index()
        c = c.drop(columns=["Date", "Url", "Text"], axis=1)
        c = c.dropna()

        try:
            c = split_ids(c, maxlen=250)
        except Exception as e:
            print(f"ERROR: Ticker {ticker}")
            print(e)
            print(c)

        # split ids sometimes generate nans
        c = c.dropna()

        # temporary
        # if mock_data then keep first segment of each article
        # and keep first segment of each date
        if mock_data:
            c = c[~c.index.duplicated(keep='first')]
            c.index = c.index.date
            c = c[~c.index.duplicated(keep='first')]
            c.index = pd.to_datetime(c.index)
            # c = c.iloc[::5, :]

        assert not c.isnull().values.any()
        context[ticker] = c

    print(context)
    return context


def get_context(contexts, tickers, date, max_text=1):
    """
    Given the pd.Dataframe of the news articles, tickers and date,
    retrieve a segment of tokens from news articles associated with
    the tickers within the date
    [:250] is associated with tickers[0] and [250] is [SEP] and [251:501] is associated with tickers[1]

    Next  = 2279
    [CLS] = 101
    [SEP] = 102

    Parameters:
    contexts (pd.Dataframe)
    date (datetime.datetime)
    tickers (List[2])
    max_text (int)

    Returns:
    ids (List[501]): tokens of a segment of news article

    """

    # if tickers[0] == "AAPL":
    #     ids = ([0] * 250) + [102] + ([1] * 250)
    # else:
    #     ids = ([1] * 250) + [102] + ([0] * 250)
    #
    # return ids

    time_string = (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    timeslice1 = contexts[tickers[0]][time_string:time_string]["Ids"]
    timeslice2 = contexts[tickers[1]][time_string:time_string]["Ids"]

    sample1 = timeslice1.sample(n=min(len(timeslice1), max_text))
    sample2 = timeslice2.sample(n=min(len(timeslice2), max_text))

    ids = []

    if not sample1.empty:
        ids += sample1.sum()
    if len(ids) < 256:
        ids += ([102] * (256 - len(ids)))

    if not sample2.empty:
        ids += sample2.sum()
    if len(ids) < 512:
        ids += ([102] * (512 - len(ids)))

    return ids


def mask_ids(ids, mask_prob):
    """
    Mask ids for bert masked language modeling
    [MASK] id is 103

    Parameters:
    ids (Tensor[batch_size, length+n_step, max_len]): tokens
    mask_prob (int): mask probability for each token

    Returns:
    ids (Tensor[batch_size, length+n_step, max_len]): tokens after removing masked elements with [MASK]
    target (Tensor[batch_size, length+n_step, max_len]): bert targets for masked elements with 0 everywhere else

    """

    max_len = len(ids[0][0])
    target = []

    for i in range(len(ids)):
        target_ = []

        for j in range(len(ids[i])):
            target__ = [0] * max_len

            for k in range(len(ids[i][j])):
                p = random.random()

                if p < mask_prob:
                    target__[k] = ids[i][j][k]
                    ids[i][j][k] = 103

            target_.append(target__)
        target.append(target_)

    return ids, target


def split_ids(dataframe, maxlen=250):
    """
    Create new rows for strings with word count exceeding maxlen
    """
    def split(ids):
        lst = [ids[i:i+maxlen] for i in range(0, len(ids), maxlen)]
        return lst

    if len(dataframe.index) > 0:
        dataframe["Ids"] = dataframe.apply(lambda row : split(row["Ids"]), axis=1)
        dataframe = dataframe.explode(["Ids"])

    return dataframe


def remove_links(dataframe):
    dataframe["Text"] = dataframe['Text'].replace(r'http\S+', '', regex=True)
    return dataframe


def remove_names(dataframe):
    dataframe["Text"] = dataframe['Text'].replace(r'@\S+', '', regex=True)
    return dataframe


def remove_duplicates(dataframe):
    dataframe = dataframe.drop_duplicates(subset=["Text"], keep=False)
    return dataframe


def preprocess(dataframe):
    """Preprocess dataframe"""
    dataframe = remove_links(dataframe)
    dataframe = remove_names(dataframe)
    dataframe = remove_duplicates(dataframe)
    return dataframe


def save_model_cloud(dir, model_name, source_name):
    """Saves the model to Google Cloud Storage"""
    bucket = storage.Client().bucket(dir)
    blob = bucket.blob(model_name)
    blob.upload_from_filename(source_name)


def save_logs_cloud(dir, logfile_name, source_name):
    """Saves log file to Google Cloud Storage"""
    bucket = storage.Client().bucket(dir)
    blob = bucket.blob(logfile_name)
    blob.upload_from_filename(source_name)

