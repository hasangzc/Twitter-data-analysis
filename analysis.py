from argparse import ArgumentParser
from collections import Counter

import numpy as np
import pandas as pd

from nrclex import NRCLex
from get_data import declareParserArguments


def DataAnalysis(df: pd.DataFrame, args: ArgumentParser):
    # Drop unnecessary columns
    df.drop("Unnamed: 0", axis=1, inplace=True)

    print("-------Check columns------------")
    print(df.columns)

    print("\n---------Top 10 users who talk the most about it---------")
    print(df["Username"].value_counts()[:10].sort_values(ascending=False))

    df["tweet_without_stopwords"] = df["tweet_without_stopwords"].astype("str")

    print("\n---------What are the top 100 words in Text column---------")
    print(Counter(" ".join(df["tweet_without_stopwords"]).split()).most_common(200))

    print(
        "\n---------How many tweets were made about this topic on the specified date?--------"
    )
    print(len(df))

    print("\n------------How many different users tweeted about this topic?--------")
    print(df["Username"].nunique())

    text_object = NRCLex(" ".join(df["tweet_without_stopwords"]))
    print("\n--------------------Sentiments using nltk------------")
    print(text_object.affect_frequencies)


if __name__ == "__main__":
    # Declare an ArgumentParser object
    parser = ArgumentParser(description="Analyze your Dataset!")
    # Add the arguments
    args = declareParserArguments(parser=parser)
    DataAnalysis(df=pd.read_csv(f"./data/after_process_{args.dataset}.csv"), args=args)
