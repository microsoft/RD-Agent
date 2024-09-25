#%%
import pandas as pd
import numpy as np
import re
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
train=pd.read_csv('/data/userdata/share/kaggle/feedback-prize-english-language-learning/train.csv')
test=pd.read_csv('/data/userdata/share/kaggle/feedback-prize-english-language-learning/test.csv')
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train['full_text'])
X_test = vectorizer.transform(test['full_text'])

X_train = pd.DataFrame.sparse.from_spmatrix(X_train)
X_test = pd.DataFrame.sparse.from_spmatrix(X_test)
#%%
y_train=train[['cohesion','syntax','vocabulary','phraseology','grammar','conventions']]

def is_sparse_df(df: pd.DataFrame) -> bool:
    # 检查 DataFrame 中的每一列是否为稀疏类型
    return any(isinstance(dtype, pd.SparseDtype) for dtype in df.dtypes)

xgb_estimator = xgb.XGBRegressor(
        n_estimators=500, random_state=0, 
        objective='reg:squarederror')

# create MultiOutputClassifier instance with XGBoost model inside
model = MultiOutputRegressor(xgb_estimator, n_jobs=2)
# model4 = XGBClassifier(early_stopping_rounds=10)

if is_sparse_df(X_train):
    X_train = X_train.sparse.to_coo()
model.fit(X_train, y_train)

if is_sparse_df(X_test):
    X_test = X_test.sparse.to_coo()
y_test = model.predict(X_test)
#%%
sample = pd.read_csv('/data/userdata/share/kaggle/feedback-prize-english-language-learning/sample_submission.csv')
sample['cohesion']=y_test[:,0]
sample['syntax']=y_test[:,1]
sample['vocabulary']=y_test[:,2]
sample['phraseology']=y_test[:,3]
sample['grammar']=y_test[:,4]
sample['conventions']=y_test[:,5]

sample['text_id']=test['text_id']

# sample.to_csv('submission.csv',index=False)
# %%
