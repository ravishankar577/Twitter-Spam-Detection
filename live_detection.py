import tweet_catch as tC
import pickle as pk
import requests
import json
from requests.structures import CaseInsensitiveDict


# twitter_client = tC.TwitterClient()
# tweets = twitter_client.get_live_feed(1)
# tweetList = []
# tweetList.append(tweets[0].text)
# accused_user = tweets[0].user.screen_name

def loadMNB():
    '''
    load classifier from disk
    '''
    with open('modelMNB.pkl', 'rb') as file:
      vectorizer, classifier = pk.load(file)
    return vectorizer, classifier

user_id = 461686577
    
url = "https://api.twitter.com/2/users/" + str(user_id) + "/tweets"
# url = "https://api.twitter.com/2/users/461686577/tweets"

headers = CaseInsensitiveDict()
headers["Authorization"] = "Bearer AAAAAAAAAAAAAAAAAAAAAPaVZQEAAAAAn2tJjw0pKqoYAy2putvO4VLIGOY%3DNohozbEFFAee5VAibEcbF7odrhzdeRVG1dLNs5pWMFCqHOwkUd"


resp = requests.get(url, headers=headers)

tweet_data = (json.loads(resp.text)['data'])
tweetList = []
for i in tweet_data:
    tweetList.append(i['text'])
# tweetList = ["#jan Idiot Chelsea Handler Diagnoses Trump With a Disease https://t.co/k8PrqcWTRI https://t.co/dRN35xtSJZ", "Eren sent a glare towards Mikasa then nodded and stood up to go help his lovely girlfriend @SincerePyrrhic. Once he arrived in the kitchenâŽ¯"]
print('live-tweets:', tweetList)
vectorizer, mnb = loadMNB()
input_transformed = vectorizer.transform(tweetList)
prediction = mnb.predict(input_transformed)

# print('Analyzed live-tweet:', tweetList)

final_output = {}
for tweet_index in range(len(tweetList)):
    tweet = tweetList[tweet_index]
    result = "spam" if prediction[tweet_index] == 1 else "not spam"
    final_output.update({tweet:result}) 
    
print("FINAL OUTPUT:", final_output)
if sum(prediction)>=3 :
    print("user is a spammer")
else:
    print("user is not a spammer")
     

# print('By user: @'+ accused_user)
# print('\nAccording to MNB Classification this tweet is', 'SPAM' if prediction else 'HAM')

