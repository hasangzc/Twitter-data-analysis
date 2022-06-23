# Import the modules
from argparse import ArgumentParser
from typing import NoReturn

import pandas as pd
import snscrape.modules.twitter as sntwitter


def declareParserArguments(parser: ArgumentParser) -> ArgumentParser:
    # Add argumets
    parser.add_argument(
        "--keywords",
        type=str,
        default="Dublin Pride",
        help="Set keywords for tweets to download",
    )

    parser.add_argument(
        "--start_date",
        type=str,
        default="2022-06-15",
        help="Determine from which date the tweets containing the specified keyword should be downloaded.",
    )

    parser.add_argument(
        "--end_date",
        type=str,
        default="2022-06-16",
        help="Determine by what date tweets containing the specified keyword should be downloaded.",
    )

    parser.add_argument(
        "--limit",
        action="store_true",
        default=False,
        help="limit the number of tweets to download",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="Dublin Pride_2022-06-15_2022-06-16",
        help="The data file name [without .csv] to be processed",
    )

    # Return the parsed arguments
    return parser.parse_args()


def get_tweets_to_csv(count_limit: int, args: ArgumentParser) -> NoReturn:
    """With this function, the desired number of tweets are captured on the specified subject and date and saved as a dataframe.
    Args:
        count_limit: limit the number of tweets to download
    Returns:
        NoReturm
    """
    # Creating list to append tweet data to
    tweets_list2 = []

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i, tweet in enumerate(
        sntwitter.TwitterSearchScraper(
            f"{args.keywords} since:{args.start_date} until:{args.end_date}"
        ).get_items()
    ):
        if args.limit:
            if i > count_limit:
                break
        tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.user.username])

    # Creating a dataframe from the tweets list above
    tweets_df2 = pd.DataFrame(
        tweets_list2, columns=["Datetime", "Tweet Id", "Text", "Username"]
    )

    tweets_df2.to_csv(f"./data/{args.keywords}_{args.start_date}_{args.end_date}.csv")


if __name__ == "__main__":
    # Declare an ArgumentParser object
    parser = ArgumentParser(
        description="Get Tweets Fetch tweets containing specified words"
    )
    # Add the arguments
    args = declareParserArguments(parser=parser)
    # For "Dublin Pride" topic, fetch 5000 tweets dated June 15-16 and save as csv.
    get_tweets_to_csv(
        args=args,
        count_limit=10,
    )
