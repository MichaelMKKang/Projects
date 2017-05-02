#This code uses a standard format for getting data from Twitter's API using tweepy.
#I based this off Marco Bonzanini's tutorial for using Tweepy. Thanks for the guidance!


#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from rq import Queue
from redis import Redis


#Variables that contains the user credentials to access Twitter API 
consumer_key = 'custom_consumer_key'
consumer_secret = 'custom_consumer_secret'
access_token = 	'custom_access_token'
access_token_secret = 'custom_access_token_secret'

#Can't get Redis to work on my laptop.
#q = Queue(connection=Redis())

#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):
    def on_data(self, data):
        print(data)
        #q.enqueue(print, data)
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: (python, health)
    stream.filter(track=['python', 'health'])
