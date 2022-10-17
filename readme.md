# DGA域名
域名生成算法（Domain Generation Algorithm， DGA） 是一项古老但一直活跃的技术， 是中心结构僵尸网络赖以生存的关键武器， 该技术给打击和关闭该类型僵尸网络造成了不小的麻烦。 研究人员需要快速掌握域名生成算法和输入， 以便对生成的域名及时进行处置。

<img src="./images/main/%E5%9F%BA%E4%BA%8EDGA%E7%9A%84%E5%83%B5%E5%B0%B8%E7%BD%91%E7%BB%9C.png">

DGA依赖时间、 字典和硬编码的常量动态生成域名。典型的DGA算法实现如下：
```
def rand(r, seed):
    return (seed - 1043968403*r) & 0x7FFFFFFF
def dga(date, seed):
    charset = string.ascii_lowercase + string.digits
    tlds = [".net", ".org", ".top"]
    unix = int(time.mktime(date.timetuple()))
    b = 7*24*3600
    c = 4*24*3600
    r = ((unix//b)*b + c)
    for i in range(200):
        domain = ""
        for _ in range(12):
            r = rand(r, seed)
            domain += charset[r % len(charset)]
        r = rand(r, seed)
        tld = tlds[r % 3]domain += tld
        print(domain)
```

# 数据集

## 白数据
Alexa是一家专门发布网站世界排名的网站， 创建于1996年4月， 以搜索引擎起家。 Alexa每天在网上搜集超过1000GB的信息， 不仅给出多达几十亿的网址链接， 而且为其中的每一个网站进行了排名。 可以说， Alexa是当前拥有URL数量最庞大、 排名信息发布最详尽的网站。 我们使用Alexa全球排名前100万的网站的域名作为白样本， 对应下载页面为：

http://www.secrepo.com/

数据下载链接:https://s3-us-west-1.amazonaws.com/umbrella-static/index.html


## 黑数据
针对DGA样本数据， 我们以360netlab的开放数据为黑样本 ， 360netlab的主页为：http://data.netlab.360.com/dga/

数据下载链接:http://data.netlab.360.com/feeds/dga/dga.txt

# 特征提取
## N-Gram模型
把域名当做一个字符串， 使用N-Gram建模， 以2-Gram为例， 把baidu.com进行建模:

<img src="./images/main/DGA%E5%9F%9F%E5%90%8D%E8%BF%9B%E8%A1%8C2-Gram%E5%A4%84%E7%90%86.png">

使用CountVectorizer进行转换即可， 其中ngram_range设置为（2， 2） ， 表明使用2-Gram， token_pattern设置为'\w'， 表明是按照字符切分：
CV = CountVectorizer(
    ngram_range=(2, 2),
    token_pattern=r'\w',
    decode_error='ignore',
    strip_accents='ascii',
    max_features=max_features,
    stop_words='english',
    max_df=1.0,
    min_df=1)
x = CV.fit_transform(x)

ngram_range设置为（2， 3） ， 表明使用2-Gram和3-Gram。 以单词baidu为例子， 处理后的结果为：
    - ba
    - ai
    - id
    - du
    - bai
    - aid
    - idu

## 统计特征模型

1. 元音字母个数
    
    正常人通常在取域名的时候， 都会偏向选取“好读”的几个字母组合， 抽象成数学可以理解的语言， 就是英文的元音字母比例会比较高。 DGA生成域名的时候， 由于时间因素是随机因素， 所以元音字母这方面的特征不明显。

    计算元音字母的比例：
    ```
    def get_aeiou(domain_list):
        x=[]
        y=[]
        for domain in domain_list:
            x.append(len(domain))
            count=len(re.findall(r'[aeiou]',domain.lower()))
            count=(0.0+count)/len(domain)
            y.append(count)
        return x,y
    ```
2. 唯一字母数字个数

    唯一的字母数字个数指的是域名中去掉重复的字母和数字后的个数， 比如：baidu唯一字母数字个数为5、facebook唯一字母数字个数为7、google唯一字母数字个数为4。
    
    唯一的字母数字个数与域名长度的比例， 从某种程度上反映了域名字符组成的统计特征。 计算唯一的字母数字个数可以使用python的set数据结构：
    ```
    def get_uniq_char_num(domain_list):
        x=[]
        y=[]
        for domain in domain_list:
            x.append(len(domain))
            count=len(set(domain))
            count=(0.0+count)/len(domain)
            y.append(count)
        return x,y

3. 平均Jarccard系数

    Jarccard系数定义为两个集合交集与并集元素个数的比值， 本例的Jarccard系数是基于2-Gram计算的。
    
    计算两个域名之间的Jarccard系数的方法为：
    ```
    def count2string_jarccard_index(a,b):
        x=set(' '+a[0])
        y=set(' '+b[0])
        for i in range(0,len(a)-1):
            x.add(a[i]+a[i+1])
        x.add(a[len(a)-1]+' ')
        for i in range(0,len(b)-1):
            y.add(b[i]+b[i+1])
        y.add(b[len(b)-1]+' ')
        return (0.0+len(x-y))/len(x|y)
    ```

    计算两个域名集合的平均Jarccard系数的方法为：
    def get_jarccard_index(a_list,b_list):
        x=[]
        y=[]
        for a in a_list:
            j=0.0
            for b in b_list:
                j+=count2string_jarccard_index(a,b)
            x.append(len(a))
            y.append(j/len(b_list))
        return x,y

## 字符序列模型

    把域名当做一个由字符组成的序列， 字符转换成对应的ASCII值， 这样就可以把域名最终转换成一个数字序列：
    t=[]
    for i in x:
        v=[]
        for j in range(0,len(i)):
            v.append(ord(i[j]))
        t.append(v)
    x=t

# 算法使用

根据兜哥《Web安全之2.深度学习》的研究，在此使用多个机器学习算法和分词模型

代码见dga.py

```
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
import re
from collections import namedtuple
from random import shuffle
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn import preprocessing
from hmmlearn import hmm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import warnings
warnings.filterwarnings("ignore")

dga_file="./data/dga/dga.txt"
alexa_file="./data/white/top-1m.csv"

def load_alexa():
    x=[]
    data = pd.read_csv(alexa_file, sep=",",header=None)
    x=[i[1] for i in data.values]
    return x

def load_dga():
    x=[]
    data = pd.read_csv(dga_file, sep="\t", header=None,
                      skiprows=18)
    x=[i[1] for i in data.values]
    return x

def get_feature_charseq():
    alexa=load_alexa()
    dga=load_dga()
    x=alexa+dga
    max_features=10000
    y=[0]*len(alexa)+[1]*len(dga)

    t=[]
    for i in x:
        v=[]
        for j in range(0,len(i)):
            v.append(ord(i[j]))
        t.append(v)

    x=t
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4)

    return x_train, x_test, y_train, y_test


def get_aeiou(domain):
    count = len(re.findall(r'[aeiou]', domain.lower()))
    #count = (0.0 + count) / len(domain)
    return count

def get_uniq_char_num(domain):
    count=len(set(domain))
    #count=(0.0+count)/len(domain)
    return count

def get_uniq_num_num(domain):
    count = len(re.findall(r'[1234567890]', domain.lower()))
    #count = (0.0 + count) / len(domain)
    return count

def get_feature():
    from sklearn import preprocessing
    alexa=load_alexa()
    dga=load_dga()
    v=alexa+dga
    y=[0]*len(alexa)+[1]*len(dga)
    x=[]

    for vv in v:
        vvv=[get_aeiou(vv),get_uniq_char_num(vv),get_uniq_num_num(vv),len(vv)]
        x.append(vvv)

    x=preprocessing.scale(x)
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4)
    return x_train, x_test, y_train, y_test

def get_feature_2gram():
    alexa=load_alexa()
    dga=load_dga()
    x=alexa+dga
    max_features=10000
    y=[0]*len(alexa)+[1]*len(dga)

    CV = CountVectorizer(
                                    ngram_range=(2, 2),
                                    token_pattern=r'\w',
                                    decode_error='ignore',
                                    strip_accents='ascii',
                                    max_features=max_features,
                                    stop_words='english',
                                    max_df=1.0,
                                    min_df=1)
    x = CV.fit_transform(x)
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4)

    return x_train.toarray(), x_test.toarray(), y_train, y_test


def get_feature_234gram():
    alexa=load_alexa()
    dga=load_dga()
    x=alexa+dga
    max_features=10000
    y=[0]*len(alexa)+[1]*len(dga)

    CV = CountVectorizer(
                                    ngram_range=(2, 4),
                                    token_pattern=r'\w',
                                    decode_error='ignore',
                                    strip_accents='ascii',
                                    max_features=max_features,
                                    stop_words='english',
                                    max_df=1.0,
                                    min_df=1)
    x = CV.fit_transform(x)
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4)

    return x_train.toarray(), x_test.toarray(), y_train, y_test

def do_nb(x_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

def do_xgboost(x_train, x_test, y_train, y_test):
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    print("text feature & nb")
    x_train, x_test, y_train, y_test = get_feature()
    do_nb(x_train, x_test, y_train, y_test)

    print("text feature & xgboost")
    x_train, x_test, y_train, y_test = get_feature()
    do_xgboost(x_train, x_test, y_train, y_test)

    print("2-gram & XGBoost")
    x_train, x_test, y_train, y_test = get_feature_2gram()
    do_xgboost(x_train, x_test, y_train, y_test)

    print("2-gram & nb")
    x_train, x_test, y_train, y_test=get_feature_2gram()
    do_nb(x_train, x_test, y_train, y_test)
