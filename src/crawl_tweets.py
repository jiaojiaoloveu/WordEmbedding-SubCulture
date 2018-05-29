from nltk.twitter import Twitter


if __name__ == '__main__':
    tw = Twitter()
    tw.tweets(to_screen=False, limit=500, repeat=True)
