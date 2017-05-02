#This code processes the data pulled a saved .txt file from using tweet_streaming.py
#Worked to get representative data by dropping irrelevant features, engineering new ones out of existing data, and cleaning text.
#Made a matrix of whether each tokenized word/word fragment was one of the 200 most common tokens.
#This was mainly done using regular expressions and NLTK.

#These functions are used in tweet_analysis.py in order to obtain features and labels for making predictive models.


import json
import pandas as pd
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
import string
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['RT', 'via', '…']

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder



#Regex used for tokenizing. Saved time by borrowed from Marco Bonzanini's tutorial on using tweepy.
#Might be worth optimizing.
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = s.split()
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

#Provides the supplementary data (aka besides the text and hashtags) such as num_followers or whether a tweet was retweeted.
def get_data(path):
    tweets_data = []
    tweets_file = open(path, "r")
    for line in tweets_file:
        try:
            tweet = json.loads(line)
            tweets_data.append(tweet)
            
        except:
            continue
    #print(count_all.most_common(5))
    return tweets_data

#Provides the text of the tweet for tokenizing and cleaning.
def get_text(path):
    tweets_text = []
    tweets_file = open(path, "r")
    count_all = Counter()
    for line in tweets_file:
        try:
            tweet = json.loads(line)
            terms_only = [term for term in preprocess(tweet['text']) if term not in stop and not term.startswith(('#', '@', 'http'))]
            count_all.update(terms_only)
            tweets_text.append(terms_only)
            
        except:
            continue
    #print(count_all.most_common(5))
    return tweets_text

#Provides the hashtags for taking frequency distributions to get the most prevalent hashtags used during data collection.
def get_hashtags(path):
    tweets_hashtags = []
    tweets_file = open(path, "r")
    count_all = Counter()
    for line in tweets_file:
        try:
            tweet = json.loads(line)
            terms_hash = [term for term in preprocess(tweet['text']) if term.startswith('#')]
            count_all.update(terms_hash)
            tweets_hashtags.append(terms_hash)
        except:
            continue
    #print(count_all.most_common(10))
    return tweets_hashtags

#Uses data from get_data() and leaves relevant features.
def make_tweets_df(path):
    tweets_data = get_data(path)
    tweets = pd.DataFrame(tweets_data)
    tweets = tweets.drop(['limit', 'place', 'possibly_sensitive', 'quoted_status', 'quoted_status_id', 'filter_level', 'truncated', 'source', 'retweeted_status', 'quoted_status_id_str', 'extended_tweet', 'extended_entities', 'display_text_range', 'contributors'], axis=1)
    tweets = tweets.drop(['in_reply_to_user_id_str', 'in_reply_to_status_id', 'in_reply_to_status_id', 'in_reply_to_user_id_str'], axis=1)
    #'in_reply_to_screen_name', 'in_reply_to_status_id', 'in_reply_screen_name', 'in_reply_to_status_id', 
    #users, text, 
    tweets = tweets.drop(['in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id', 'is_quote_status', 'geo'], axis=1)
    tweets = tweets.drop(['favorite_count', 'retweet_count', 'retweeted', 'id_str', 'id', 'favorited', 'coordinates'], axis=1)
    
    #Creating column on what time tweet was created (probably should make into ordinal number on how long since I started recording)
        #Droping feature for now since it doesn't seem to correlate much with tweet usage (within the short amount of time (hours) that I'm gathering data for)
        #Might be interesting to look further into to see how the usage of a particular hashtag grows/falls and how to plug that into algorithm
    #tweets['time_created_at'] = pd.to_datetime(tweets['created_at'], format='%a %b %d  %H:%M:%S +%f %Y')

    #Should drop 'entities' if getting hashtags from get_hashtags() function
    tweets = tweets.drop(['created_at', 'timestamp_ms', 'entities'], axis=1)
    #Dropping tweets without any text
    tweets = tweets[tweets.text.notnull()]
    #Feature engineering if texts are in mostly english or not.
    tweets['eng_lang'] = tweets['lang']
    tweets.loc[tweets.eng_lang != 'en', 'eng_lang'] = 0
    tweets.loc[tweets.eng_lang == 'en', 'eng_lang'] = 1
    tweets = tweets.drop('lang', axis=1)
    return tweets

#Takes the text of each tweet, processes it using NLTK and regular expressions, then combines it with data from make_tweets_df into
#   a single pandas dataframe.
def make_data_df(path):
    #Get text-based information
    text = get_text(path)
    birds = []
    for words in text:
        words_filtered = [w.lower() for w in words if len(w)>=3]
        birds.append(words_filtered)
    #Take top features, with cutoff being that token is present in at least 0.05% of tweets (or the top 200 most frequent tokens)
    words = []
    for blurbs in birds:
        words.extend(blurbs)
    #Getting cleaning words starting with apostrophes, words that are quote/unquoted, semicolons, colons, elipses
    for i in range(0,len(words)):
        if words[i].startswith(("'",'"','…', ':', '.', ',')):
            words[i] = words[i][1:]
        if words[i].endswith(("'",'"','…', ':', '.', ',', ';')): #These two will turn elipses into a just a period
            words[i] = words[i][:-1]

    #Getting frequency of words.
    words = nltk.FreqDist(words)
    mostcommon = words.most_common(200)                        #Total length of 40726 tweets. 
    common_words = []
    for terms in mostcommon:
        common_words.append(terms[0])
    
    features = []
    for i in range(0,len(birds)):               #for each set of words from each domainname
        word = set(birds[i])
        temp = {}
        for feat in common_words:              #for each token from the word_features
            temp[feat] = (feat in word)
        features.append(dict(temp))
        
    #Get information out of tweets.
    tweets = make_tweets_df(path)
    tweet_starts = tweets.text.tolist()
    RT_list = [0]*len(tweet_starts)
    at_list = [0]*len(tweet_starts)
    for i in range(0,len(tweet_starts)):
        if tweet_starts[i].startswith('RT'):
            RT_list[i]=1
            tweet_starts[i] = tweet_starts[i][3:]
        if tweet_starts[i].startswith('@'):
            at_list[i]=1
            
    #Putting it all together into a final dataframe
    data = pd.DataFrame(features)
    data['eng_lang'] = tweets['eng_lang']
    data['user'] = tweets['user']
    #data['time_created_at'] = tweets['time_created_at']
    #adding new features of whether or not retweeted and whether or not @ someone else
    data['Retweeted'] = pd.Series(at_list)
    data['@someone_else'] = pd.Series(at_list)
    data = process_users(data)
    #We find that 31 tweets have their eng_lang feature missing. Filling with the most common 'en'
    data['eng_lang'] = data['eng_lang'].fillna(1)
    return data

#Returns supplementary information about the twitter accounts that produced each tweet.
def process_users(data):
    user_list = data['user'].tolist()
    favourites_count = []
    followers_count = []
    friends_count = []
    time_zone = []
    statuses_count = []
    for i in range(0,len(user_list)):
        try:
            favourites_count.append(user_list[i]['followers_count'])
        except TypeError:
            favourites_count.append(0)
        
    for i in range(0,len(user_list)):
        try:
            followers_count.append(user_list[i]['followers_count'])
        except TypeError:
            followers_count.append(0)
           
    for i in range(0,len(user_list)):
        try:
            friends_count.append(user_list[i]['friends_count'])
        except TypeError:
            friends_count.append(0)
        
    for i in range(0,len(user_list)):
        try:
            statuses_count.append(user_list[i]['statuses_count'])
        except TypeError:
            statuses_count.append(0)
        
    for i in range(0,len(user_list)):
        try:
            time_zone.append(user_list[i]['time_zone'])
        except TypeError:
            time_zone.append('Not specified')
    times = time_zone                                       #Some of the elements in the list are still None. Replacing...
    for i in range(0,len(time_zone)):
        if time_zone[i] == None:
            times[i] = 'Not specified'
       
    data['favourites_count'] = pd.Series(favourites_count)
    data['followers_count'] = pd.Series(followers_count)
    data['friends_count'] = pd.Series(friends_count)
    data['statuses_count'] = pd.Series(statuses_count)
    zone = LabelEncoder().fit_transform(times)                        #Or use LabelBinarizer?
    data['time_zone'] = pd.Series(zone)
    #Converting lang into ordinal. Since vast majority is in english, it should be fine for now? Will experiment to see if worth changing
    #data['language'] = LabelBinarizer().fit_transform(data['lang'].factorize())
    #Dropping user column
    data = data.drop(['user'], axis=1)
    return data

#Takes hashtags and returns an array of whether each tweet had a hashtag within the most common hashtags seen in the data provided.
def make_labels(path):
    htags = get_hashtags(path)
    tgs = []
    for tags in htags:
        tags_filtered = [h.lower() for h in tags]
        tgs.append(tags_filtered)
    hsh = []
    for tag in tgs:
       hsh.extend(tag)
    #Cleaning hashtags get rid of punctuation
    for i in range(0,len(hsh)):
        if len(hsh[i]) > 2:                 #To avoid getting rid of #…
            if hsh[i].endswith((':', '…', '!')):
                hsh[i] = hsh[i][:-1]

    #Getting frequency of hashtags
    hsh = nltk.FreqDist(hsh)
    mostcommonhash = hsh.most_common(40)        #at least 75 instances of each tweet
    common_hash = []
    for terms in mostcommonhash:
        common_hash.append(terms[0])

    results = []
    for i in range(0,len(tgs)):                                         #For each row
        rowtags = []
        for j in range(0,len(tgs[i])):                                  #For each hashtag in each row
            indivhash = []
            for k in range(0, len(common_hash)):                        #For each common_hashtag
                if common_hash[k] in tgs[i][j]:                         #If common_hashtag equals hashtag (basically) [Should this be 'in' instad of '=='?] (look at this again once i get rid of characters in the hashtags)
                    indivhash.append(common_hash[k])
            if len(indivhash) != 0:
                rowtags.append(indivhash)
        results.append(rowtags)                                   #Will soon figure out how to not have the hashtags in lists of lists of lists

    labels = [[]]*len(results)
    for i in range(0,len(results)):
        if len(results[i]) !=0:
            labels[i] = results[i][0]
        else:
            labels[i] = ['none']
    Y = MultiLabelBinarizer().fit_transform(labels)
    return Y
