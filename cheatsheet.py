
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

#读数据
df = pd.read_csv('')

#数据基本情况
df.shape
df.head()
df.columns
df.describe()
df.info()
df.dtypes

#特征间相关性检查
corr = df.corr()
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')
corr[""].sort_values(ascending=False)

#查看空值情况
df.isnull().sum()

#查看各个值的情况
df.age.value_counts()

#画出数据集的所有直方图矩阵
df.hist(bins=50,figsize=(20,15))

#画各个数据频率的直方图
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True, palette='Set2')
plt.title('Distribution of age')
plt.xlabel('age')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()

#查看选定变量之间的关系矩阵
from pandas.plotting import scatter_matrix
attr = ["medv","rm","zn","b","ptratio","lstat"]
scatter_matrix(df[attr],figsize=(12,8))

#画散点图
df.plot(kind = "scatter", x="rm", y ="medv")
#画直方图
df.plot(kind = "scatter", x="rm", y ="medv")

#拆分数据
X = df.drop(["medv"],axis=1)
y = df['medv']

#选择特定几个特征
X_new = df[['rm','lstat']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#标准化：使用StandardScaler来标准化数据，使其均值为0，方差为1。
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)

#归一化：使用MinMaxScaler将数据缩放到特定范围（通常是0到1）。
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

#线性回归训练
from sklearn import linear_model
clf = linear_model.LinearRegression()#定义线性模型
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(score)

#交叉验证
from sklearn.model_selection import cross_val_score 
scores = cross_val_score(linear_model.LinearRegression(), X_test, y_test, cv=5)
print("Cross-validation scores:", scores)

#文本清洗
from gensim.parsing import strip_tags, strip_numeric, strip_multiple_whitespaces, stem_text, strip_punctuation, remove_stopwords
from gensim.parsing import preprocess_string
import re
# 将字符串转换为小写的lambda函数
transform_to_lower = lambda s: s.lower()
# 移除单字符（用正则表达式）的lambda函数
remove_single_char = lambda s: re.sub(r'\s+\w{1}\s+', '', s)

# 定义一个名为 CLEAN_FILTERS 的列表，包含一组预处理过滤器。这些过滤器按顺序执行，依次对文本进行处理。过滤器包括：
# strip_tags: 移除 HTML 标签。
# strip_numeric: 移除数字。
# strip_punctuation: 移除标点符号。
# strip_multiple_whitespaces: 移除多余的空白字符。
# transform_to_lower: 转换为小写。
# remove_stopwords: 移除停用词。
# remove_single_char: 移除单字符单词。
CLEAN_FILTERS = [strip_tags,
                strip_numeric,
                strip_punctuation, 
                strip_multiple_whitespaces, 
                transform_to_lower,
                remove_stopwords,
                remove_single_char]

#定义一个清洗pipline方法
def cleaning_pipe(document):
    processed_words = preprocess_string(document, CLEAN_FILTERS)
    return processed_words

# Apply the cleaning pipe on the news data

pd_df['clean_text'] = pd_df['news'].apply(cleaning_pipe)