from google.cloud import storage
import pandas as pd
import random
import json
import os


def read_sp500(filename="data/sp500.csv"):
    file = pd.read_csv(filename)
    tickers = file["Symbol"].tolist()
    return tickers


def read_tickers():
    path = f"C:/PycharmProjects/stockai/data/context/news/"
    tickers = os.listdir(path)
    tickers = filter_tickers(tickers)
    return tickers


def read_bert_config(model_type):
    path = f"pretrained_model/{model_type}_config.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data


def filter_tickers(tickers):
    """
    if context news data incomplete then filter out ticker
    (bad data)
    """
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
    file = pd.read_csv(filename)
    tickers = file["Symbol"].tolist()
    names = file["Description"].tolist()

    res = {tickers[i]: names[i].replace(" Inc", "").replace(" Corp", "")
           for i in range(len(tickers))}
    return res


def read_prices(tickers, start, end, repeat=1):
    prices = []
    for ticker in tickers:
        price = pd.read_csv(f"/home/yh04/PycharmProjects/stockai/data/prices/{ticker}.csv")
        price["Date"] = pd.to_datetime(price["Date"])
        price.set_index("Date", inplace=True)

        prices.append(price)

    prices = pd.concat(prices, axis=1, keys=tickers)
    prices = prices.iloc[:, prices.columns.get_level_values(1) == "Close"]
    prices = prices.loc[start:end]

    prices = prices.loc[prices.index.repeat(repeat)]

    return prices


def read_context_file(filename):
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
    :return: A dictionary with tickers as keys and dataframe of texts as value
    """
    context = dict()
    srcdir = "/home/yh04/PycharmProjects/stockai/data/context"
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
    :param contexts: pd.Dataframe
    :param date:     datetime.datetime
    :param tickers:  List[2]
    :param max_text: int
    :return:         List[501]

    "Next"  = 2279
    "[CLS]" = 101
    "[SEP]" = 102
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
    if len(ids) < 250:
        ids += ([102] * (250 - len(ids)))

    ids += [102]

    if not sample2.empty:
        ids += sample2.sum()
    if len(ids) < 501:
        ids += ([102] * (501 - len(ids)))

    return ids


def mask_ids(ids, mask_prob):
    """
    :param ids:       [batch_size, length+n_step, max_len]
    :param mask_prob: probability of masking each word
    :return:
    """

    target = []

    for i in range(len(ids)):
        target_ = []

        for j in range(len(ids[i])):
            target__ = [0] * 501

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
    :return: Create new rows for strings with word count exceeding maxlen
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

