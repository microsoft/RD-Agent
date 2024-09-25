import pandas as pd
import numpy as np
import re
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import xgboost as xgb

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

import pandas as pd
train=pd.read_csv('../input/feedback-prize-english-language-learning/train.csv')
test=pd.read_csv('../input/feedback-prize-english-language-learning/test.csv')

def data_cleaner(text):
    text = text.strip()
    text = re.sub(r'\n', '', text)
    text = text.lower()
    return text

train['full_text']=train['full_text'].apply(data_cleaner)
test['full_text']=test['full_text'].apply(data_cleaner)


import nltk
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
def generate_sentiment_scores(data):
    sid = SentimentIntensityAnalyzer()
    neg=[]
    pos=[]
    neu=[]
    comp=[]
    for sentence in tqdm(data['full_text'].values): 
        sentence_sentiment_score = sid.polarity_scores(sentence)
        comp.append(sentence_sentiment_score['compound'])
        neg.append(sentence_sentiment_score['neg'])
        pos.append(sentence_sentiment_score['pos'])
        neu.append(sentence_sentiment_score['neu'])
    return comp,neg,pos,neu
train['compound'],train['negative'],train['positive'],train['neutral']=generate_sentiment_scores(train)
test['compound'],test['negative'],test['positive'],test['neutral']=generate_sentiment_scores(test)


train['com_len']=train['full_text'].apply(lambda x:len(x.split()))
test['com_len']=test['full_text'].apply(lambda x:len(x.split()))


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train['full_text'])
X_test = vectorizer.transform(test['full_text'])


from sklearn.preprocessing import Normalizer
trans = Normalizer()
X_train_com=trans.fit_transform(train['compound'].values.reshape(-1,1))
X_test_com=trans.transform(test['compound'].values.reshape(-1,1))


from sklearn.preprocessing import Normalizer
trans = Normalizer()
X_train_neg=trans.fit_transform(train['negative'].values.reshape(-1,1))
X_test_neg=trans.transform(test['negative'].values.reshape(-1,1))


from sklearn.preprocessing import Normalizer
trans = Normalizer()
X_train_pos=trans.fit_transform(train['positive'].values.reshape(-1,1))
X_test_pos=trans.transform(test['positive'].values.reshape(-1,1))


from sklearn.preprocessing import Normalizer
trans = Normalizer()
X_train_neu=trans.fit_transform(train['neutral'].values.reshape(-1,1))
X_test_neu=trans.transform(test['neutral'].values.reshape(-1,1))


from sklearn.preprocessing import Normalizer
trans = Normalizer()
X_train_len=trans.fit_transform(train['com_len'].values.reshape(-1,1))
X_test_len=trans.transform(test['com_len'].values.reshape(-1,1))


from scipy.sparse import hstack
train_s=hstack((X_train,X_train_com,X_train_neg,X_train_pos,X_train_neu,X_train_len))
test_s=hstack((X_test,X_test_com,X_test_neg,X_test_pos,X_test_neu,X_test_len))

y=train[['cohesion','syntax','vocabulary','phraseology','grammar','conventions']]

params_lgb = {
    "n_estimators": 1000,
    "verbose": -1
}


y_train=train[['cohesion','syntax','vocabulary','phraseology','grammar','conventions']]

xgb_estimator = xgb.XGBRegressor(
        n_estimators=500, random_state=0, 
        objective='reg:squarederror')

# create MultiOutputClassifier instance with XGBoost model inside
model4 = MultiOutputRegressor(xgb_estimator, n_jobs=2)
# model4 = XGBClassifier(early_stopping_rounds=10)
model4.fit(train_s, y_train)

y_test = model4.predict(train_s)

sample = pd.read_csv('../input/feedback-prize-english-language-learning/sample_submission.csv')
sample['cohesion']=y_test[:,0]
sample['syntax']=y_test[:,1]
sample['vocabulary']=y_test[:,2]
sample['phraseology']=y_test[:,3]
sample['grammar']=y_test[:,4]
sample['conventions']=y_test[:,5]

sample['text_id']=test['text_id']

sample.to_csv('submission.csv',index=False)