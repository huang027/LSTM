from keras.preprocessing import sequence
from keras.optimizers import SGD,RMSprop,Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU
import jieba
import pandas as pd
import numpy as np
from sklearn import metrics
#读取训练语料
neg=pd.read_excel('G:\\python_work\\neg_1.xls',header=None,index_col=None)
pos=pd.read_excel('G:\\python_work\\pos_1.xls',header=None,index_col=None)
#给训练语料贴上标签
pos['mark']=1
neg['mark']=0
pn=pd.concat([pos,neg],ignore_index=True)#合并语料
neglen=len(neg)
poslen=len(pos)#计算语料数目
cw=lambda x:list(jieba.cut(x)) #定义分词函数
pn['words']=pn[0].apply(cw)
comment=pd.read_excel('G:\\python_work\\sum_1.xls')#读入评论内容
comment=comment[comment['rateContent'].notnull()] #仅读取非空评论
comment['words']=comment['rateContent'].apply(cw) #评论分词
d2v_train=pd.concat([pn['words'],comment['words']],ignore_index=True)

w=[]#将所有词语整合在一起
for i in d2v_train:
    w.extend(i)
dict=pd.DataFrame(pd.Series(w).value_counts())#统计词的出现次数

del w,d2v_train
dict['id']=list(range(1,len(dict)+1))
get_sent=lambda x:list(dict['id'][x])
pn['sent']=pn['words'].apply(get_sent)
print(pn)
print(dict)

maxlen=50
print("pad sequences(samples x time)")
pn['sent']=list(sequence.pad_sequences(pn['sent'],maxlen=maxlen))
#训练集
x=np.array(list(pn['sent']))[::2]
y=np.array(list(pn['mark']))[::2]
#测试集
xt=np.array(list(pn['sent']))[1::2]
yt=np.array(list(pn['mark']))[1::2]
#全集
xa=np.array(list(pn['sent']))
ya=np.array(list(pn['mark']))
model=Sequential()
model.add(Embedding(len(dict)+1,256,input_length=maxlen))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')
model.fit(x,y,batch_size=16,nb_epoch=10) #训练时间较长
classes=model.predict_classes(xt)
acc=metrics.accuracy_score(yt,classes)
print('test accuracy',acc)




