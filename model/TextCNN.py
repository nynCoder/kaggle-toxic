import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.preprocessing import text, sequence#文本与序列预处理模块
from keras.callbacks import Callback

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'
#使用网上预训练的词向量下载
EMBEDDING_FILE = r'D:\BaiduNetdiskDownload\有毒评论\wiki-news-300d-1M-subword.vec\wiki-news-300d-1M-subword.vec'
train = pd.read_csv(r"D:\BaiduNetdiskDownload\有毒评论\toxic\train.csv")
test = pd.read_csv(r"D:\BaiduNetdiskDownload\有毒评论\toxic\test.csv")
print(train.shape,test.shape)
# submission = pd.read_csv(r"D:\BaiduNetdiskDownload\有毒评论\toxic\sample_submission.csv")
X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values
max_features=100000#word_size，即字典数/特征数
maxlen=200#每句话的最大长度
embd_size=300#词向量：300维
#这个类用来对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示。
#init(num_words) 构造函数，传入词典的最大值
tokenizer=text.Tokenizer(num_words=max_features)
#将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
'''Tokenizer是如何判断文本的一个词呢？其实它是以空格去识别每个词。因为英文的词与词之间是以空格分隔，
所以我们可以直接将文本作为函数的参数，但是当我们处理中文文本时，我们需要使用分词工具将词与词分开，
并且词间使用空格分开。'''
tokenizer.fit_on_texts(list(X_train)+list(X_test))
#text_to_word_sequence(text,fileter) 可以简单理解此函数功能类str.split,分割
X_train=tokenizer.texts_to_sequences(X_train)
X_test=tokenizer.texts_to_sequences(X_test)
#不够的填充，多了的截断
x_train=sequence.pad_sequences(X_train,maxlen=maxlen)
x_test=sequence.pad_sequences(X_test,maxlen=maxlen)
def get_coefs(word,arr):
    #词，词向量
    return word,np.asarray(arr,dtype='float32')
embedding_index=dict(get_coefs(o.rstrip().split(' ')[0],o.rstrip().split(' ')[1:])for o in open(EMBEDDING_FILE,encoding='utf8'))#这里我们使用预训练的词向量
word_index=tokenizer.word_index# 一个dict，保存所有word对应的编号id，从1开始
nb_words=min(max_features,len(word_index))#调整字典的大小，节省空间
#初始化一个词向量矩阵作为权重矩阵,通过神经网络的训练迭代更新得到一个合适的权重矩阵
embedding_matrix=np.zeros((nb_words,embd_size))
'''embedding的本质是一个字典，每一个词以及它对应的词向量。而往往会想，这么多词，怎么运用到自己的训练集上呢？
答案是我们针对自己的训练集建立自己的字典，然后依据字典去提取我们需要的词向量进行了，
然后就可以建立我们的词向量矩阵，
'''
for word,i in word_index.items():
    if i>=max_features:
        continue
    embedding_vector=embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector


#评估模型
class RocAucEvaluation(Callback):
    def __init__(self,validation_data=(),interval=1):
        super(Callback,self).__init__()
        self.interval=interval
        self.X_val,self.y_val=validation_data
    def on_epoch_end(self,epoch,logs={}):
        if epoch%self.interval==0:
            y_pred=self.model.perdict(self.X_val,verbose=0)
            score=roc_auc_score(self.y_val,y_pred)
            print("\nROC-AUC -epoch:%d -score:%.6f \n"%(epoch+1,score))
# 卷积核大小和个数
filters = [1, 2, 3, 5]
num_filters = 32


# 对应那个图，完美！！
# keras语法之小括号:第一个括号，是构建一个层，第二个括号，用该层做计算
def get_model():
    inp = Input(shape=(maxlen,))  # 这里输入的其实是每句话里的词的id，之后通过embedding_matrix找到对应的向量映射
    # embedding层的作用，是将正整数下标转换为具有固定大小的向量。
    # keras中没有使用乘法（虽然本质是乘法），而是直接使用的通过下标去查找映射，所以维度很容易让人迷惑！
    x = Embedding(max_features, embd_size, weights=[embedding_matrix])(inp)
    # 可以预训练的word embedding+随机初始化char embedding进行拼接
    x = SpatialDropout1D(0.4)(x)
    x = Reshape((maxlen, embd_size, 1))(x)
    '''自然语言是一维数据, 虽然经过word-embedding 生成了二维向量，但是对词向量做从左到右滑动来进行卷积没有意义. 
    比如 "今天" 对应的向量[0, 0, 0, 0, 1], 按窗口大小为 1* 2 从左到右滑动得到[0,0], [0,0], [0,0], [0, 1]这四个向量, 
    对应的都是"今天"这个词汇, 这种滑动没有帮助.'''
    # 卷积，针对不同卷积核大小（词的覆盖度）注意文本分类中卷积核宽度为卷积多少单词，长度一般为词向量维度，即水平方向没有滑动，只是从上向下滑动。
    conv_0 = Conv2D(num_filters, kernel_size=(filters[0], embd_size), kernel_initializer='normal', activation='elu')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filters[1], embd_size), kernel_initializer='normal', activation='elu')(x)
    conv_2 = Conv2D(num_filters, kernel_size=(filters[2], embd_size), kernel_initializer='normal', activation='elu')(x)
    conv_3 = Conv2D(num_filters, kernel_size=(filters[3], embd_size), kernel_initializer='normal', activation='elu')(x)
    '''卷积层输入的是一个表示句子的矩阵，维度为n*d，即每句话共有n个词，每个词有一个d维的词向量表示。假设Xi:i+j表示Xi到Xi+j个词，使用一个宽度为d，高度为h的卷积核W与Xi:i+h-1(h个词)
    进行卷积操作后再使用激活函数激活得到相应的特征ci，则卷积操作可以表示为：（使用点乘来表示卷积操作）
   因此经过卷积操作之后，可以得到一个n-h+1维的向量'''
    # 例如输出形状为input-filter_size+1:200-1+1,200-2+1,200-3+1,200-5+1....
    # 池化：池化使用1-max pooling。

    '''filter_size=1,2，3，5. 每个filter 的宽度与词向量等宽，这样只能进行一维滑动。
    每一种filter卷积后，结果输出为[batch_size, seq_length - filter_size +1,1,num_filter]的tensor。
    由于我们有三种filter_size, 故会得到三种tensor'''
    maxpool_0 = MaxPool2D(pool_size=(maxlen - filters[0] + 1, 1))(conv_0)  # ---》卷积后维度200-1+1=200（batch,200,1，32）
    maxpool_1 = MaxPool2D(pool_size=(maxlen - filters[1] + 1, 1))(conv_1)  # ---》卷积后维度200-2+1=199（batch,199,1，32）
    maxpool_2 = MaxPool2D(pool_size=(maxlen - filters[2] + 1, 1))(conv_2)  # ---》卷积后维度200-3+1=198（batch,198,1,32）
    maxpool_3 = MaxPool2D(pool_size=(maxlen - filters[3] + 1, 1))(conv_3)  # ---》卷积后维度200-5+1=196（batch,196,1,32）
    # 最大池化【每个卷积核都要取最大的那个，然后拼接，这里是32】每个都是（batch,1，32），最后拼接
    '''max-pooling 在保持主要特征的情况下, 大大降低了参数的数目, 从图五中可以看出 feature map 从 三维变成了一维, 好处有如下两点: 

降低了过拟合的风险, feature map = [1, 1, 2] 或者[1, 0, 2] 最后的输出都是[2], 表明开始的输入即使有轻微变形, 也不影响最后的识别。

参数减少, 进一步加速计算。pooling 本身无法带来平移不变性(图片有个字母A, 这个字母A 无论出现在图片的哪个位置, 在CNN的网络中都可以识别出来)，
卷积核的权值共享才能做到. max-pooling的原理主要是从多个值中取一个最大值，做不到这一点。cnn 能够做到平移不变性，是因为在滑动卷积核的时候，
使用的卷积核权值是保持固定的(权值共享), 假设这个卷积核被训练的就能识别字母A, 当这个卷积核在整张图片上滑动的时候，
当然可以把整张图片的A都识别出来。'''
    # 列向连接
    # 这里也可以【maxpool,avgpool】拼接试试效果
    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])  # --》拼接【batch,1，1，32*4】
    z = Flatten()(z)  # --》变形一下，【batch,128】为了后边的全连接层
    z = Dropout(0.1)(z)
    # sigmoid ==> 标签之间概率独立，所有Pred值的和也不再为1，softmax ==> 标签之间概率不独立,归一化为概率值
    # 因为是多标签分类，所以激活函数用sigmoid，做了6个二分类记住！！！！！！！！
    outp = Dense(6, activation="sigmoid")(z)  # 【batch,6】
    model = Model(inputs=inp, outputs=outp)
    # 编译
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

    #     print(model.summary())  #查看模型网络结构
    return model


model = get_model()
model=get_model()
batch_size=256
epochs=3
X_tra,X_val,y_tra,y_val=train_test_split(x_train,y_train,train_size=0.95,random_state=233)
RocAuc=RocAucEvaluation(validation_data=(X_val,y_val),interval=1)
model.fit(X_tra,y_tra,batch_size=batch_size,epochs=epochs,validation_data=(X_val,y_val),callbacks=[RocAuc],verbose=2)
y_pred=model.predict(x_test,batch_size=1024)
#提交结果文件
submission =pd.DataFrame(y_pred,columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
# submission.to_csv(r'D:\BaiduNetdiskDownload\有毒评论\toxic\submission.csv', index=False)

