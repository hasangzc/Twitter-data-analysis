import warnings
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px

from wordcloud import WordCloud
from get_data import declareParserArguments


def count_sentiments(df: pd.DataFrame, column_name: str):
    # Make direction for result plot and dataframe
    Path(f"./visualization_results/").mkdir(parents=True, exist_ok=True)
    sns.countplot(df[column_name])
    # Save resulting plot
    plt.savefig(
        f"./visualization_results/count_sentiments.png",
        bbox_inches="tight",
    )


def most_common_20_words(df: pd.DataFrame, column_name: str):
    # convert column datatype to string
    df[column_name] = df[column_name].astype("str")
    # Count most common words without stopwords
    data = pd.DataFrame(Counter(" ".join(df[column_name]).split()).most_common(20))
    # Determine columns
    data.columns = ["Common_words", "count"]
    # Plot it
    fig = px.bar(
        data,
        x="count",
        y="Common_words",
        title="Commmon Words in Selected Text",
        orientation="h",
        width=700,
        height=700,
        color="Common_words",
    )
    # Save resulting plot as html format
    fig.write_html(f"./visualization_results/common_words.html")


def positive_words(df: pd.DataFrame):
    # Determine figure sizes
    plt.figure(figsize=(15, 15))
    # Create work cloud object
    wc = WordCloud(max_words=2000, width=1600, height=800).generate(
        " ".join(df[df.Sentiment == "Positive"].lemmatized_text)
    )
    plt.imshow(wc, interpolation="bilinear")
    plt.savefig(f"./visualization_results/positive_words.png")


def negative_words(df: pd.DataFrame):
    # Determine fig sizes
    plt.figure(figsize=(15, 15))
    # Create an workcloud object
    wc = WordCloud(max_words=2000, width=1600, height=800).generate(
        " ".join(df[df.Sentiment == "Negative"].lemmatized_text)
    )
    plt.imshow(wc, interpolation="bilinear")
    plt.savefig(f"./visualization_results/negative_words.png")


def visualize(args: ArgumentParser):
    # count sentiments
    count_sentiments(
        df=pd.read_csv(f"./data/after_process_{args.dataset}.csv"),
        column_name="Sentiment",
    )

    most_common_20_words(
        df=pd.read_csv(f"./data/after_process_{args.dataset}.csv"),
        column_name="tweet_without_stopwords",
    )

    positive_words(df=pd.read_csv(f"./data/after_process_{args.dataset}.csv"))

    negative_words(df=pd.read_csv(f"./data/after_process_{args.dataset}.csv"))


if __name__ == "__main__":
    parser = ArgumentParser(description="visualizations")
    args = declareParserArguments(parser=parser)
    # Ignore warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)
    visualize(args=args)
