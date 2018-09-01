import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from imblearn.under_sampling import RandomUnderSampler

##数据预处理
offline=pd.read_csv('F:/天池大赛/offline_final_data_without_na.csv')
del offline['Unnamed: 0']
offline_data=offline[['Distance','Coupon_type','discount_rate','man','jian']]
offline_target=offline['type']
#进行欠采用处理，使得样本分布更均匀
model_RandomUnderSampler = RandomUnderSampler()
data, target =model_RandomUnderSampler.fit_sample(offline_data,offline_target)
#返回的是narray，需要转换成dataframe
data=pd.DataFrame(data,columns=['Distance','Coupon_type','discount_rate','man','jian'])
target=pd.Series(target)    #这里必须是series，不然后面的包无法识别
#54138个正负样本，正样本都保留了
#建模
xgb=XGBClassifier(booster='gbtree',n_estimators=100,max_depth=8,gamma=0.1,objective='binary:logistic')
#交叉验证
scores=cross_val_score(estimator=xgb,X=data,y=target,cv=5,scoring='roc_auc')
print('roc_auc:%.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
#学习曲线
train_sizes, train_scores, test_scores = learning_curve(estimator=xgb, X=data, y=target, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0, 1.0])
plt.show()

#画出的学习曲线表明，没有明显的过拟合，稍微有点欠拟合，两条线收敛于0.78

##调参
xgb=XGBClassifier(booster='gbtree',n_estimators=100,max_depth=10,gamma=0,learning_rate=0.01,objective='binary:logistic')
scores=cross_val_score(estimator=xgb,X=data,y=target,cv=5,scoring='roc_auc')
print('roc_auc:%.3f +/- %.3f' % (np.mean(scores),np.std(scores)))

#对测试集进行预测
#先也要对测试集进行同训练集一样的转换，重新写函数用apply来进行
pre=pd.read_csv('F:/天池大赛/ccf_offline_stage1_test_revised.csv')
del pre['Merchant_id']
pre.set_index(['User_id','Coupon_id','Date_received'],inplace = True)

pre['Coupon_type']=None
pre['discount_rate']=None
pre['man']=None
pre['jian']=None


import re

def trans(x):
    if ':' in x['Discount_rate']:
        m=re.match(r'(.*):(.*)', x['Discount_rate'])
        x['Coupon_type']=1
        x['man'] = m.group(1)
        x['jian']= m.group(2)
    else:
        x['Coupon_type']=0
        x['discount_rate']=x['Discount_rate']
    return x

pre=pre.apply(trans,axis=1)

del pre['Discount_rate']

pre['discount_rate']=pre['discount_rate'].fillna(0)
pre['Distance']=pre['Distance'].fillna(pre['Distance'].mean())
test=pre[['Distance','Coupon_type','discount_rate','man','jian']]
test[['man','jian']]=test[['man','jian']].fillna(0)

xgb_model=xgb.fit(data,target)
test=test.astype(float)
result=xgb_model.predict_proba(test)

test['result']=np.around(result,1)[:,1]  #注意round函数在python3中是5舍6入，numpy带的around是4舍5入

test=test.reset_index()

final.to_csv('F:/天池大赛/final_2018_7_31.csv',index=False)
test.to_csv('F:/天池大赛/test_2018_7_31.csv')

