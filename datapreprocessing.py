from argparse import ArgumentParser

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from get_data import declareParserArguments


def TwitterDataPipeline(df: pd.DataFrame, args: ArgumentParser) -> pd.DataFrame:
    # Change data type of time column
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Remove Url's in dataset
    df["Text"] = df["Text"].str.replace(r"http\S+", "", regex=True)

    # Remove @username expression in dataset
    df["Text"] = df["Text"].str.replace(r"@[^\s]+", "", regex=True)

    # Check null values
    print("---------Check null values-----------")
    if df.isna().sum().sum() > 0:
        print(
            "There are nan values in your dataset! Check columns and remove nan values;"
        )
        print(df.isna().sum())
        # Drop nan values in dataset
        df = df.dropna()
    else:
        print("There are no nan values in your dataset!\n")

    # Drop unnecessary columns
    df.drop("Unnamed: 0", axis=1, inplace=True)

    # Converting strings to a lower case in Text column
    df["Text"] = df["Text"].str.lower()

    # Remove stop words in tokenized_text column
    stop = stopwords.words("english")
    df["tweet_without_stopwords"] = df["Text"].apply(
        lambda x: " ".join([word for word in x.split() if word not in (stop)])
    )

    # Tokenize Text column
    df["tokenized_text"] = df["tweet_without_stopwords"].apply(word_tokenize)

    # Lemmatization for tokenized_text column
    lmtzr = WordNetLemmatizer()
    df["lemmatized_text"] = df["tokenized_text"].apply(
        lambda lst: [lmtzr.lemmatize(word) for word in lst]
    )

    # sort columns
    df = df[
        [
            "Datetime",
            "Tweet Id",
            "Username",
            "Text",
            "tweet_without_stopwords",
            "tokenized_text",
            "lemmatized_text",
        ]
    ]

    df["lemmatized_text"] = df["lemmatized_text"].astype("str")

    df["lemmatized_text"] = df["lemmatized_text"].str.replace("[", "", regex=True)
    df["lemmatized_text"] = df["lemmatized_text"].str.replace("]", "", regex=True)
    df["lemmatized_text"] = df["lemmatized_text"].str.replace("'", "", regex=True)
    df["lemmatized_text"] = df["lemmatized_text"].str.replace(",", "", regex=True)

    nltk.download("vader_lexicon")
    sentiments = SentimentIntensityAnalyzer()
    df["Positive"] = [
        sentiments.polarity_scores(i)["pos"] for i in df["lemmatized_text"]
    ]
    df["Negative"] = [
        sentiments.polarity_scores(i)["neg"] for i in df["lemmatized_text"]
    ]
    df["Neutral"] = [
        sentiments.polarity_scores(i)["neu"] for i in df["lemmatized_text"]
    ]
    df["Compound"] = [
        sentiments.polarity_scores(i)["compound"] for i in df["lemmatized_text"]
    ]

    score = df["Compound"].values
    sentiment = []
    for i in score:
        if i >= 0.05:
            sentiment.append("Positive")
        elif i <= -0.05:
            sentiment.append("Negative")
        else:
            sentiment.append("Neutral")
    df["Sentiment"] = sentiment

    # Save the prepared data
    df.to_csv(
        f"./data/after_process_{args.keywords}_{args.start_date}_{args.end_date}.csv"
    )


if __name__ == "__main__":
    # Declare an ArgumentParser object
    parser = ArgumentParser(description="Clean your Dataset!")
    # Add the arguments
    args = declareParserArguments(parser=parser)
    # Process to data
    TwitterDataPipeline(df=pd.read_csv(f"./data/{args.dataset}.csv"), args=args)


# helper: https://www.kaggle.com/code/ruchi798/sentiment-analysis-the-simpsons/notebook
