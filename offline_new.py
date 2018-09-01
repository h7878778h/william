import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from imblearn.under_sampling import RandomUnderSampler
import re
##导入数据并融合
offline=pd.read_csv('F:/天池大赛/ccf_offline_stage1_train/ccf_offline_stage1_train.csv')
#online=pd.read_csv('F:/天池大赛/ccf_online_stage1_train/ccf_online_stage1_train.csv')
offline_consumer_features=pd.read_csv('F:/天池大赛/offline_consumer_features.csv')
offline_consumer_features_use_rates=pd.read_csv('F:/天池大赛/offline_consumer_features_use_rates.csv')
offline_consumer_merchant_features=pd.read_csv('F:/天池大赛/offline_consumer_merchant_features.csv')
offline_merchant_features=pd.read_csv('F:/天池大赛/offline_merchant_features.csv')
offline_merchant_features_use_rates=pd.read_csv('F:/天池大赛/offline_merchant_features_use_rates.csv')
#融合上述表
raw_data=pd.merge(offline,offline_consumer_features,on=['User_id']) #merge貌似一次只能融合2个表
raw_data=pd.merge(raw_data,offline_consumer_features_use_rates,on=['User_id'])
raw_data=pd.merge(raw_data,offline_merchant_features,on=['Merchant_id'])
raw_data=pd.merge(raw_data,offline_consumer_merchant_features,on=['User_id','Merchant_id'])
raw_data=pd.merge(raw_data,offline_merchant_features_use_rates,on=['Merchant_id'])
#dtypes查看，除了折扣率都是数字型（int or float）
raw_data.to_csv('F:/天池大赛/raw_data.csv',index=False)


##处理转换数据，打标签
#日期列转换成日期格式

raw_data=pd.read_csv('F:/天池大赛/raw_data.csv')

raw_data[['Date','Date_received']]=raw_data[['Date','Date_received']].fillna(0,axis=1)
raw_data['Date']=raw_data['Date'].astype(int)
raw_data['Date']=raw_data['Date'].astype(str)
raw_data['Date_received']=raw_data['Date_received'].astype(int)
raw_data['Date_received']=raw_data['Date_received'].astype(str)
raw_data['Date']=pd.to_datetime(raw_data['Date'],format='%Y%m%d',errors = 'coerce')
raw_data['Date_received']=pd.to_datetime(raw_data['Date_received'],format='%Y%m%d',errors = 'coerce')
#删除没有领券日期的行（没领券）
raw_data=raw_data.dropna(subset=['Date_received'])

#对折扣进行转换
len(raw_data['Discount_rate'].unique())   #有45种折扣
raw_data['Discount_rate']=raw_data['Discount_rate'].astype(str)
#增加列
raw_data['Coupon_type']=None
raw_data['discount_rate']=None
raw_data['man']=None
raw_data['jian']=None

#apply感觉也很慢，而且不清楚其源码，总是有问题。还是用原有迭代吧
#def m(x):
#    if ':' in x:
#        m = re.match(r'(.*):(.*)', x)
#        y=1
#        z = m.group(1)
#        f= m.group(2)
#        h=0
#    else:
#        y = 0
#        z=0
#        f=0
#        h=x
#    return pd.Series([y,z,f,h], index=['Coupon_type','man','jian','discount_rate'])
#raw_data[['Coupon_type','man','jian','discount_rate']]=raw_data.apply(lambda x:m(x['Discount_rate']),axis=1)

for index,series in raw_data.iterrows():
    if ':' in series['Discount_rate']:
        m = re.match(r'(.*):(.*)', series['Discount_rate'])
        raw_data.loc[index,'Coupon_type']=1
        raw_data.loc[index,'man'] = m.group(1)
        raw_data.loc[index,'jian']= m.group(2)
    else:
        raw_data.loc[index, 'Coupon_type'] = 0
        raw_data.loc[index,'discount_rate']= series['Discount_rate']
#跑了一天一夜还没完，先停止输出为文件'F:/天池大赛/raw_data_2018-8-9.csv'

#data取raw_data的前50行试验该方法是否可行
def m(x):
    if ':' in x:
        m = re.match(r'(.*):(.*)', x)
        y=1
        z = m.group(1)
        f= m.group(2)
        h=0
    else:
        y = 0
        z=0
        f=0
        h=x
    return pd.Series([y,z,f,h], index=['Coupon_type','man','jian','discount_rate'])

raw_data[['Coupon_type','man','jian','discount_rate']]=raw_data.apply(lambda x:m(x['Discount_rate']),axis=1)

#一晚上就跑完了，确实快很多。保存为‘raw_data_8_10_完整’

#打分类标签
def f(x,y):
    if (y-x).days<=15:
        return 1
    else:return 0

raw_data['type']=raw_data.apply(lambda x:f(x['Date_received'],x['Date']),axis=1)

data=raw_data[['Distance','num_consume_x','num_coupon_receive_x','num_use_coupon_x','coupon_use_rate_x','num_consume_y','num_coupon_receive_y','num_use_coupon_y','num_consume','num_use_coupon','coupon_use_rate_y','Coupon_type','discount_rate','man','jian']]

data['Distance']=data['Distance'].fillna(5) #用中位数进行填充，认为其不远不近，并且从算法上讲使得NA的影响较小。

type=raw_data['type']

#试一下决策树
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=5,class_weight='balanced')

from sklearn.model_selection import cross_val_score
from sklearn import metrics
scores=cross_val_score(tree,offline_data,offline_target,cv=10,scoring='roc_auc')
print('roc_auc:%.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
#auc:0.932，roc显著提升，但此步没有平衡样本

##xgboost
raw_data=pd.read_csv('F:/天池大赛/测试集.csv')
data=raw_data[['Distance','num_consume_x','num_coupon_receive_x','num_use_coupon_x','coupon_use_rate_x','num_consume_y','num_coupon_receive_y','num_use_coupon_y','num_consume','num_use_coupon','coupon_use_rate_y','Coupon_type','discount_rate','man','jian']]
data['Distance']=data['Distance'].fillna(5)
target=raw_data['type']
#平衡样本
model_RandomUnderSampler = RandomUnderSampler()
data_balance, target_banlance =model_RandomUnderSampler.fit_sample(data,target)
data_balance=pd.DataFrame(data_balance,columns=['Distance','num_consume_x','num_coupon_receive_x','num_use_coupon_x','coupon_use_rate_x','num_consume_y','num_coupon_receive_y','num_use_coupon_y','num_consume','num_use_coupon','coupon_use_rate_y','Coupon_type','discount_rate','man','jian'])
target_banlance=pd.Series(target_banlance)

#建模
xgb=XGBClassifier(booster='gbtree',n_estimators=100,max_depth=8,gamma=0.1,objective='binary:logistic')
scores=cross_val_score(estimator=xgb,X=data_balance,y=target_banlance,cv=5,scoring='roc_auc')
print('roc_auc:%.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
#auc:0.970

#画学习曲线
train_sizes, train_scores, test_scores = learning_curve(estimator=xgb, X=data_balance, y=target_banlance, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0, 1.0])
plt.show()
#有些过拟合

#调参
from sklearn.learning_curve import validation_curve
param_range=[10,20,30,40,50,60,70,80,90,100]
train_scores_validation_curve,test_scores_validation_curve=validation_curve(estimator=xgb,X=data_balance,y=target_banlance,param_name='n_estimators',param_range=param_range,cv=5)
#作图
train_mean_vc=np.mean(train_scores_validation_curve,axis=1)
train_std_vc=np.std(train_scores_validation_curve,axis=1)
test_mean_vc=np.mean(test_scores_validation_curve,axis=1)
test_std_vc=np.std(test_scores_validation_curve,axis=1)
plt.plot(param_range,train_mean_vc,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(param_range,train_mean_vc+train_std_vc,train_mean_vc-train_std_vc,alpha=0.15,color='blue')
plt.plot(param_range,test_mean_vc,color='green',linestyle='--',marker='s',markersize=5,label='validation accuracy')
plt.fill_between(param_range,test_mean_vc+test_std_vc,test_mean_vc-test_std_vc,alpha=0.15,color='green')
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.5,1.0])
plt.show()
#结果表明，n_estimators（树的颗数）越低越好。
xgb=XGBClassifier(booster='gbtree',n_estimators=10,max_depth=5,gamma=0.1,objective='binary:logistic')
scores=cross_val_score(estimator=xgb,X=data_balance,y=target_banlance,cv=5,scoring='roc_auc')
print('roc_auc:%.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
#auc:0.977
xgb=XGBClassifier(booster='gbtree',n_estimators=5,max_depth=5,gamma=0.1,objective='binary:logistic')
scores=cross_val_score(estimator=xgb,X=data_balance,y=target_banlance,cv=5,scoring='roc_auc')
print('roc_auc:%.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
#auc:0.977

##测试集特征工程
test_raw=pd.read_csv('F:/天池大赛/ccf_offline_stage1_test_revised.csv')

test['Coupon_type']=None
test['discount_rate']=None
test['man']=None
test['jian']=None

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
test=test.apply(trans,axis=1)

test['discount_rate']=test['discount_rate'].fillna(0)
test['Distance']=test['Distance'].fillna(5)
test[['man','jian']]=test[['man','jian']].fillna(0)
#导入表
offline_consumer_features=pd.read_csv('F:/天池大赛/offline_consumer_features.csv')
offline_consumer_features_use_rates=pd.read_csv('F:/天池大赛/offline_consumer_features_use_rates.csv')
offline_consumer_merchant_features=pd.read_csv('F:/天池大赛/offline_consumer_merchant_features.csv')
offline_merchant_features=pd.read_csv('F:/天池大赛/offline_merchant_features.csv')
offline_merchant_features_use_rates=pd.read_csv('F:/天池大赛/offline_merchant_features_use_rates.csv')
#融合上述表，采用左连接，因为test中有些数据在其他的表中没有
raw_test=pd.merge(test,offline_consumer_features,how='left',on=['User_id'])
raw_test=pd.merge(raw_test,offline_consumer_features_use_rates,how='left',on=['User_id'])
raw_test=pd.merge(raw_test,offline_merchant_features,how='left',on=['Merchant_id'])
raw_test=pd.merge(raw_test,offline_consumer_merchant_features,how='left',on=['User_id','Merchant_id'])
raw_test=pd.merge(raw_test,offline_merchant_features_use_rates,how='left',on=['Merchant_id'])
#填补空白值,都用中位数填充试试
raw_test['num_consume_x']=raw_test['num_consume_x'].fillna(raw_test['num_consume_x'].median())
raw_test['num_coupon_receive_x']=raw_test['num_coupon_receive_x'].fillna(raw_test['num_coupon_receive_x'].median())
raw_test['num_use_coupon_x']=raw_test['num_use_coupon_x'].fillna(raw_test['num_use_coupon_x'].median())
raw_test['coupon_use_rate_x']=raw_test['coupon_use_rate_x'].fillna(raw_test['coupon_use_rate_x'].median())
raw_test['num_coupon_receive_y']=raw_test['num_coupon_receive_y'].fillna(raw_test['num_coupon_receive_y'].median())
raw_test['num_use_coupon_y']=raw_test['num_use_coupon_y'].fillna(raw_test['num_use_coupon_y'].median())
raw_test['num_consume']=raw_test['num_consume'].fillna(raw_test['num_consume'].median())
raw_test['num_use_coupon']=raw_test['num_use_coupon'].fillna(raw_test['num_use_coupon'].median())
raw_test['coupon_use_rate_y']=raw_test['coupon_use_rate_y'].fillna(raw_test['coupon_use_rate_y'].median())

#预测
xgb=XGBClassifier(booster='gbtree',n_estimators=10,max_depth=5,gamma=0.1,objective='binary:logistic')
xgb_model=xgb.fit(data_balance,target_banlance)
test=raw_test[['Distance','num_consume_x','num_coupon_receive_x','num_use_coupon_x','coupon_use_rate_x','num_consume_y','num_coupon_receive_y','num_use_coupon_y','num_consume','num_use_coupon','coupon_use_rate_y','Coupon_type','discount_rate','man','jian']]
test=test.astype(float)
result=xgb_model.predict_proba(test)

test_raw['result']=np.around(result,1)[:,1]  #注意round函数在python3中是5舍6入，numpy带的around是4舍5入

predict=test_raw[['User_id','Coupon_id','Date_received','result']]
predict.to_csv('F:/天池大赛/final_2018_8_10.csv',index=False)