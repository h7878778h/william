import numpy as np
import pandas as pd

offline=pd.read_csv('F:/天池大赛/ccf_offline_stage1_train/ccf_offline_stage1_train.csv')
online=pd.read_csv('F:/天池大赛/ccf_online_stage1_train/ccf_online_stage1_train.csv')

###数据初探

off_group=offline.groupby(offline['User_id'])   #将offline的数据按照user_id进行分组
off=off_group.count()   #分组后进行count计算（复制到Ipython中执行才能直接得到out）分组后User_id这一栏直接变成了index！！
#offline数据有：1754884条id，539438条唯一id（合并重复项）
on_group=online.groupby(online['User_id'])   #将online的数据按照user_id进行分组，分组后User_id这一栏直接变成了index！！
on=on_group.count()
#online数据又：11429826条id，762858条唯一id

#因为上述操作后user_id没有了，变成了index。所以重新新建一个列‘User_id’，使其等于index
on['User_id']=on.index
off['User_id']=off.index
on_off=pd.merge(on,off,on='User_id')   #类似于mysql的join，按照user_id相同进行融合
#得到online数据和offline数据有267448个共有id，保存在变量on_off中

#计算商户merchant_id 有多少个唯一值
off_group_merchant=offline.groupby(['Merchant_id'],as_index=False)  # as_index=False 使得列不会变成索引。参考上面代码
off_merchant=off_group_merchant.count()
#off：8415个商户id

on_group_merchant=online.groupby(['Merchant_id'],as_index=False)
on_merchant=on_group_merchant.count()
#on：7999个商户id

#看on和off数据中商户的交集
on_off_merchant=pd.merge(off_merchant,on_merchant,on='Merchant_id')
#得到该dataframe是空，说明二者完全没有交集，与参考的文档一致。

#绘制线下购买量的时间序列图

offline_date=offline.dropna(subset=['Date'])    #删除Date含有NA值的行
offline_date['Date']=offline_date['Date'].astype(int)     #该dataframe的类型为float，为了后续转换成date方便，将其转换成int整数 或者 object 字符串类型

#将字符串解析成datetime对象：offline_date['Date']=pd.to_datetime(offline_date['Date'])   to_datetime不能解析该格式的字符格式

off_date=offline_date.groupby(['Date'],as_index=False)
off_date1=off_date.count()  #看每个date对应各列的 count计数，即可得到每个日期对应的交易、领券等情况

#绘图
off_date1=off_date1.set_index(['Date'])     #将Date设置为index，因为作图函数是以index为x坐标的
off_date1['User_id','Merchant_id'].plot()
off.date1.to_csv('C:/Users/Administrator/Desktop/off_date.csv')     #发现用plot函数画图，若没有将date转换成时间格式则图很难看。所以将该数据导出用excel作图。


###建模预测：此次是为了入门练手，熟悉如何导入数据、初探数据、处理数据，如何调用算法来建模，所以此次就不进行特征工程了，直接利用线下数据来建模，（可以考虑删除2月4号到2月23号之间的数据，因为从初探结果来看，该时间段是春节，其线下数据对7月预测来说不具参考价值）
###offline提供的数据足够多，可以暂时不用online的数据了
###思路：1.将日期数据都想办法转换成dataframe可以识别的DATE数据，方便后续处理
### 2.先划分有效数据，因为要预测领券后15天内是否消费，所以最后15天领券的数据考虑不用，即6月15日之后领券的数据删除
### 3.领券日期为null的（没领券）的数据删除（没法分别其实正类还是负类）
### 4.将id都变成索引，避免在训练模型时将其当做一个变量运算影响结果。
### 5.折扣如何表示:共有45种折扣（包括满减和直接打折），可以对其进行one-hot编码，但特征空间会变得比较大，且很稀疏。考虑对其进行转换：0表示为满减，1表示为折扣，再增加其他列表示满的门槛，另一列表示减的额度，再一列表示折扣率
### 6.打标签，分正负类样本
###
###
###
###
###
###
###
###
import datetime
import time


offline=pd.read_csv('F:/天池大赛/ccf_offline_stage1_train/ccf_offline_stage1_train.csv')

#offline先填充date,dete_received两列的na为0，否则无法转换成str，转换前去掉小数点，先转为int
offline[['Date','Date_received']]=offline[['Date','Date_received']].fillna(0,axis=1)
offline['Date']=offline['Date'].astype(int)
offline['Date']=offline['Date'].astype(str)
offline['Date_received']=offline['Date_received'].astype(int)
offline['Date_received']=offline['Date_received'].astype(str)
offline['Date']=pd.to_datetime(offline['Date'],format='%Y%m%d',errors = 'coerce') #用to_datetime方法将该列变成date类型数据。errors = 'coerce',If ‘coerce’, then invalid parsing will be set as NaT
offline['Date_received']=pd.to_datetime(offline['Date_received'],format='%Y%m%d',errors = 'coerce')
offline=offline.dropna(subset=['Date_received'])
offline=offline.set_index('Date_received',drop=False)   #领券日期建立索引,共有1053282行
offline=offline[:'2016-6-16']   #通过切片索引，仅保留2016-6-16日之前领券的数据，发现其实共1053282行都在这个日期范围内。

#对折扣进行转换
len(offline['Discount_rate'].unique())   #有45种折扣
offline['Discount_rate']=offline['Discount_rate'].astype(str)
offline.reset_index(drop=True,inplace=True)  #将索引重新变成1~n的整数索引，因为用日期做索引有重复值，这不利于后续处理
#增加列
offline['Coupon_type']=None #表示折扣券种类，是满减（1）还是直接折扣（0）
offline['discount_rate']=None   #折扣率
offline['man']=None #满多少
offline['jian']=None    #减多少

#对dataframe进行修改
import re

for index,series in offline.iterrows(): #dataframe的iterrows()方法能产生一个生成器，返回index和index对应的内容（为series结构）
    if ':' in series['Discount_rate']:
        m = re.match(r'(.*):(.*)', series['Discount_rate'])
        offline.loc[index,'Coupon_type']=1
        offline.loc[index,'man'] = m.group(1)
        offline.loc[index,'jian']= m.group(2)
    else:
        offline.loc[index, 'Coupon_type'] = 0
        offline.loc[index,'discount_rate']= series['Discount_rate']

#先导入时间序列相关模块，用不用得上再说
import datetime
import time

#先将两列都转换成datetime格式
offline['Date']=pd.to_datetime(offline['Date'],format='%Y-%m-%d',errors = 'coerce')
offline['Date_received']=pd.to_datetime(offline['Date_received'],format='%Y-%m-%d',errors = 'coerce')

#apply函数历遍dataframe的一种方式！
def f(x,y):
    if (y-x).days<=15:
        return 1
    else:return 0

offline['type']=offline.apply(lambda x:f(x['Date_received'],x['Date']),axis=1)
#共98W+负例，6w+正例


##利用决策树构建模型
#不能有NA。先对offline['discount_rate'].fillna(0),再offline.dropna()
offline=pd.read_csv('F:/天池大赛/offline_final_data_without_na.csv')

offline_data=offline[['Distance','Coupon_type','discount_rate','man','jian']]   #特征集
offline_target=offline['type']  #标签

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(offline_data, offline_target, test_size=0.3)   #x_train 和 y_train构成机器学习的训练集，_test构成验证集

from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=5,class_weight='balanced')
tree_result=tree.fit(x_train,y_train)


#导出训练得到的决策树
import sklearn
import pydotplus

featute_name=['Distance','Coupon_type','discount_rate','man','jian']
target_name=['0','1']
dot_data = sklearn.tree.export_graphviz(tree_result, out_file=None,
                         feature_names=featute_name,
                         class_names=target_name,
                         filled=True, rounded=True,
                         special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('F:/天池大赛/tree1.pdf')

#对模型进行评估
from sklearn import cross_validation,metrics
predict = tree_result.predict(x_test)
test_auc = metrics.roc_auc_score(y_test,predict)
#AUC:0.72....


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
pre['Distance']=pre['Distance'].fillna(pre['Distance'].mean())  #用均值填充distan的空值

test=pre[['Distance','Coupon_type','discount_rate','man','jian']]
test[['man','jian']]=test[['man','jian']].fillna(0)
result=tree_result.predict_proba(test)

test['result']=np.around(result,1)[:,1]  #注意round函数在python3中是5舍6入，numpy带的around是4舍5入

test=test.reset_index()

final=test[['User_id','Coupon_id','Date_received','result']]
final.to_csv('F:/天池大赛/final_2018_7_17.csv',index=False)


##决策树建模，k折交叉验证
import pandas as pd
import numpy as np

offline=pd.read_csv('F:/天池大赛/offline_final_data_without_na.csv')
#分出特征集和标签
offline_data=offline[['Distance','Coupon_type','discount_rate','man','jian']]   #特征集
offline_target=offline['type']  #标签
#导入评估模型
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=5,class_weight='balanced')

from sklearn.model_selection import cross_val_score
from sklearn import metrics
#scores保存k折交叉验证后的各个score，最后算其均值和方差既可得到k折交叉验证后的结果
scores=cross_val_score(tree,offline_data,offline_target,cv=10,scoring='roc_auc') #当cv 指定为int 类型时，默认使用KFold 或StratifiedKFold 进行数据集打乱，下面会对KFold 和StratifiedKFold 进行介绍。
print('roc_auc:%.3f +/- %.3f' % (np.mean(scores),np.std(scores)))   #得到交叉验证ROC_auc的均值和波动范围
#0.784 +/- 0.004

#绘制学习曲线
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator=tree, X=offline_data, y=offline_target, train_sizes=np.linspace(0.1, 1.0, 10), cv=10)
#train_size需要输入一个数组（包含训练集的不同数量，相当于x轴），如果是小数，则表示是总数的百分比，如果是整数，就表示绝对数量。这里用了numpy的linspace方法来生成这样一种等差数组）
#learning_curve方法返回三个结果
train_mean = np.mean(train_scores, axis=1)
#train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
#test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0, 1.0])
plt.show()
#认为模型欠拟合

#调参：用基尼系数来选择特征（CART算法）
tree=DecisionTreeClassifier(criterion='gini',splitter='best')
scores=cross_val_score(estimator=tree,X=offline_data,y=offline_target,cv=10,scoring='roc_auc') #当cv 指定为int 类型时，默认使用KFold 或StratifiedKFold 进行数据集打乱，下面会对KFold 和StratifiedKFold 进行介绍。
print('roc_auc:%.3f +/- %.3f' % (np.mean(scores),np.std(scores)))   #得到交叉验证ROC_auc的均值和波动范围

###使用逻辑回归
import pandas as pd
import numpy as np

offline=pd.read_csv('F:/天池大赛/offline_final_data_without_na.csv')
#分出特征集和标签
offline_data=offline[['Distance','Coupon_type','discount_rate','man','jian']]   #特征集
offline_target=offline['type']  #标签
#进行特征缩放
from sklearn.preprocessing import StandardScaler
#offline_data=offline_data.values #转换成numpy array形式，才能用这个模块.实践证明不一定，但返回的是array形式
offline_data=StandardScaler().fit_transform(offline_data)   #返回成了array形式，需要将其重新弄成dataframe
offline_data=offline_data.T #转置方便弄成dataframe
c={'Distance':offline_data[0],'Coupon_type':offline_data[1],'discount_rate':offline_data[2],'man':offline_data[3],'jian':offline_data[4]}
offline_data=pd.DataFrame(c)
#导入评估模型
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(class_weight='balanced',max_iter=10000)
from sklearn.cross_validation import cross_val_score
scores=cross_val_score(estimator=lr,X=offline_data,y=offline_target,cv=10,scoring='roc_auc')
print('roc_auc:%.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
#绘制学习曲线
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator=lr, X=offline_data, y=offline_target, train_sizes=np.linspace(0.1, 1.0, 10), cv=10)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.5, 1.0])
plt.show()

##发现样本不均衡问题很验证，需要处理
#查看有多少不均衡样本
offline['type'].value_counts()  #86W+ 负例，54138正例

##SVM：很慢
import pandas as pd
import numpy as np

offline=pd.read_csv('F:/天池大赛/offline_final_data_without_na.csv')
offline_data=offline[['Distance','Coupon_type','discount_rate','man','jian']]
offline_target=offline['type']

from sklearn.preprocessing import StandardScaler
offline_data=StandardScaler().fit_transform(offline_data)
offline_data=offline_data.T
c={'Distance':offline_data[0],'Coupon_type':offline_data[1],'discount_rate':offline_data[2],'man':offline_data[3],'jian':offline_data[4]}
offline_data=pd.DataFrame(c)

from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
svm=SVC(kernel='rbf',class_weight='balanced')
scores=cross_val_score(estimator=svm,X=offline_data,y=offline_target,cv=10,scoring='roc_auc')
print('roc_auc:%.3f +/- %.3f' % (np.mean(scores),np.std(scores)))

#绘制学习曲线
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator=lr, X=offline_data, y=offline_target, train_sizes=np.linspace(0.1, 1.0, 10), cv=10)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.5, 1.0])
plt.show()


##朴素贝叶斯,适用于文本分类
import pandas as pd
import numpy as np

offline=pd.read_csv('F:/天池大赛/offline_final_data_without_na.csv')
offline_data=offline[['Distance','Coupon_type','discount_rate','man','jian']]
offline_target=offline['type']

from sklearn.preprocessing import StandardScaler
offline_data=StandardScaler().fit_transform(offline_data)
offline_data=offline_data.T
c={'Distance':offline_data[0],'Coupon_type':offline_data[1],'discount_rate':offline_data[2],'man':offline_data[3],'jian':offline_data[4]}
offline_data=pd.DataFrame(c)

from sklearn.naive_bayes import
from sklearn.cross_validation import cross_val_score
bayes=
scores=cross_val_score(estimator=svm,X=offline_data,y=offline_target,cv=10,scoring='roc_auc')
print('roc_auc:%.3f +/- %.3f' % (np.mean(scores),np.std(scores)))

#绘制学习曲线
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator=lr, X=offline_data, y=offline_target, train_sizes=np.linspace(0.1, 1.0, 10), cv=10)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.5, 1.0])
plt.show()


##KNN,慢！
import pandas as pd
import numpy as np

offline=pd.read_csv('F:/天池大赛/offline_final_data_without_na.csv')
offline_data=offline[['Distance','Coupon_type','discount_rate','man','jian']]
offline_target=offline['type']

from sklearn.preprocessing import StandardScaler
offline_data=StandardScaler().fit_transform(offline_data)
offline_data=offline_data.T
c={'Distance':offline_data[0],'Coupon_type':offline_data[1],'discount_rate':offline_data[2],'man':offline_data[3],'jian':offline_data[4]}
offline_data=pd.DataFrame(c)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
knn=KNeighborsClassifier(n_neighbors=5,algorithm='auto',p=2)
scores=cross_val_score(estimator=knn,X=offline_data,y=offline_target,cv=10,scoring='roc_auc')
print('roc_auc:%.3f +/- %.3f' % (np.mean(scores),np.std(scores)))

#绘制学习曲线
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator=lr, X=offline_data, y=offline_target, train_sizes=np.linspace(0.1, 1.0, 10), cv=10)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.5, 1.0])
plt.show()

##随机森林
import pandas as pd
import numpy as np

offline=pd.read_csv('F:/天池大赛/offline_final_data_without_na.csv')
offline_data=offline[['Distance','Coupon_type','discount_rate','man','jian']]
offline_target=offline['type']


from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics
RF=RandomForestClassifier(n_estimators=10,criterion='gini',oob_score=True)
RF_result=RF.fit(offline_data,offline_target) #score 对数据进行训练后输出得分（准确率）
#这里oob_score选择True，则是用“袋外数据”进行了交叉验证，相当于这个包自己就有交叉验证的功能，所以直接输出其得分即可。

y_predprob = RF_result.predict_proba(offline_data)[:,1]
print ("AUC Score (Train): %f" % metrics.roc_auc_score(offline_target,y_predprob))
#输出AUC，0.788429

###xgboost
import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn import cross_validation, metrics

offline=pd.read_csv('F:/天池大赛/offline_final_data_without_na.csv')
offline_data=offline[['Distance','Coupon_type','discount_rate','man','jian']]
offline_target=offline['type']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(offline_data, offline_target, test_size=0.3)

train = xgb.DMatrix(x_train, y_train)
test= xgb.DMatrix(x_test,y_test)

param = {'booster': 'gbtree','gamma': 0,'objective':'binary:logistic','max_depth': 6, 'eta': 0.01, 'silent': 1, 'objective': 'binary:logistic','nthread':2,'lambda': 2,'silent': 1,'scale_pos_weight':4,'eval_metric':'auc','eta':0.02}

plst = params.items()

num_rounds = 10 #应该是树的棵数
model = xgb.train(plst, train, num_rounds)  #训练数据

pre_test = model.predict(test)  #预测

###可以用xgboost中的sklearn 接口,之前作废
import pandas as pd
import numpy as np

from sklearn import cross_validation, metrics

offline=pd.read_csv('F:/天池大赛/offline_final_data_without_na.csv')
offline_data=offline[['Distance','Coupon_type','discount_rate','man','jian']]
offline_target=offline['type']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(offline_data, offline_target, test_size=0.3)

from xgboost.sklearn import XGBClassifier

clf=XGBClassifier(booster='gbtree',n_estimators=100,max_depth=8,gamma=0,scale_pos_weight=1,objective='binary:logistic')

model_sklearn=clf.fit(x_train, y_train)
y_pre = model_sklearn.predict(x_test)
test_auc = metrics.roc_auc_score(y_test,y_pre)
#只有0.5+

#交叉验证看看
from sklearn.cross_validation import cross_val_score
scores=cross_val_score(estimator=clf,X=offline_data,y=offline_target,cv=5,scoring='roc_auc')
print('roc_auc:%.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
#auc:0.78
#绘制学习曲线
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator=clf, X=offline_data, y=offline_target, train_sizes=np.linspace(0.1, 1.0, 10), cv=10)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0, 1.0])
plt.show()

##突然意识到一个问题，原始训练集中应该存在有大量完全重复的数据。
#去掉重复
import pandas as pd
import numpy as np

from sklearn import cross_validation, metrics

offline=pd.read_csv('F:/天池大赛/offline_final_data_without_na.csv')
del offline['Unnamed: 0']
offline_drop_dup=offline.drop_duplicates(keep='first')
#90多万条数据变得仅有656
offline_data=offline_drop_dup[['Distance','Coupon_type','discount_rate','man','jian']]
offline_target=offline_drop_dup['type']

from xgboost.sklearn import XGBClassifier

xgb=XGBClassifier(booster='gbtree',n_estimators=50,max_depth=10,gamma=0.1,scale_pos_weight=1.2,objective='binary:logistic')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(offline_data, offline_target, test_size=0.3)


model_sklearn=xgb.fit(x_train, y_train)
y_pre = model_sklearn.predict(x_test)
test_auc = metrics.roc_auc_score(y_test,y_pre)


#交叉验证看看
from sklearn.cross_validation import cross_val_score
scores=cross_val_score(estimator=xgb,X=offline_data,y=offline_target,cv=10,scoring='roc_auc')
print('roc_auc:%.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
#auc:0.135!搞不明白，为什么感觉像是类别标签弄反了。
#解释：感觉应该是，直接删除重复后，得到的样本有很多是相互矛盾的，比如说某些样本特征相同，但类别却不一样，所以导致模型分类出错的概率更大，也就是说auc小于0.5！
#绘制学习曲线
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator=xgb, X=offline_data, y=offline_target, train_sizes=np.linspace(0.1, 1.0, 10), cv=10)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0, 1.0])
plt.show()

#绘制验证曲线,看决策树最好的max_depth超参
from sklearn.learning_curve import validation_curve
param_range=[1,2,3,4,5,6,7,8,9,10]
train_scores_validation_curve,test_scores_validation_curve=validation_curve(estimator=tree,X=data,y=target,param_name='max_depth',param_range=param_range,cv=10)
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
plt.ylim([0.6,0.8])
plt.show()
#结果表明，5最好

##网格搜索法寻找最优超参组合
from sklearn.grid_search import GridSearchCV
