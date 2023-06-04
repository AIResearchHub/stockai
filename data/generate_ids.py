import os
import pandas as pd

from pytorch_pretrained_bert import BertTokenizer
import logging
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)


tokenizer = BertTokenizer.from_pretrained('bert_transformer-base-uncased', cache_dir="pretrained_bert")


def convert_to_ids(text):
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    return indexed_tokens


def get_files(tickers):
    files = []

    for ticker in tickers:
        newspath = f"context/news/{ticker}"
        if os.path.exists(newspath):
            for file in os.listdir(newspath):
                files.append(f"{newspath}/{file}")
        tweetspath = f"context/news/{ticker}"
        if os.path.exists(tweetspath):
            for file in os.listdir(tweetspath):
                files.append(f"{tweetspath}/{file}")

    return files


def generate_ids(tickers):
    files = get_files(tickers)

    for file in files:
        df = pd.read_json(file, lines=True)
        if len(df) > 0 and "Ids" not in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df["Date"] = df.index

            # Generate Ids column
            df["Ids"] = df.apply(lambda row : convert_to_ids(row["Text"]), axis=1)

            print(df)
            if len(df) != 0: df.to_json(file, orient="records", lines=True)


if __name__ == "__main__":
    from pytorch_pretrained_bert import BertTokenizer
    from utils import read_sp500

    tickers = read_sp500("sp500.csv")
    generate_ids(tickers)


