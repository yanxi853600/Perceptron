# Perceptron (線性可分) 訓練iris資料集
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import iris data
iris = datasets.load_iris()
x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
print("target_names: "+str(iris['target_names']))
y = pd.DataFrame(iris['target'], columns=['target'])
iris_data = pd.concat([x,y], axis=1)
#print(iris_data.head(3))

#選出花萼長度、花瓣長度、花的種類
#print(iris['target_names'])
target_name = {
    0:'setosa',1:'versicolor',2:'virginica'
}
iris_data['target_name'] = iris_data['target'].map(target_name)
iris_data = iris_data[(iris_data['target_name'] == 'setosa')|(iris_data['target_name'] == 'versicolor')]
iris_data = iris_data[['sepal length (cm)','petal length (cm)','target_name']]
#print(iris_data.head(5))

#把target_name欄位兩種花改成 1 -1
target_class = {
    'setosa':1,
    'versicolor':-1
}

iris_data['target_class'] = iris_data['target_name'].map(target_class)

del iris_data['target_name']
#print(iris_data.head(5))


#激勵函數 sign
def sign(z):
    if z > 0:
        return 1
    else:
        return -1


#初始化w =[0,0,0] (會自動更新調整)
        
w = np.array([0.,0.,0.])
error = 1 #主要是紀錄沒有錯誤分類的話就停止_線性可分
iterator = 0 #記錄更新了幾次
while error != 0:
    error = 0
    
    #迴圈一筆一筆去跑iris裡面的資料
    for i in range(len(iris_data)):
        
        #把x的資料都加上x0=1
        x,y = np.concatenate((np.array([1.]), np.array(iris_data.iloc[i])[:2])), np.array(iris_data.iloc[i])[2]
        
        #指預測失敗時
        if sign(np.dot(w,x)) != y:
            print("iterator: "+str(iterator))
            iterator += 1
            error += 1
            sns.lmplot('sepal length (cm)','petal length (cm)',data=iris_data, fit_reg=False, hue ='target_class')
            
            # 前一個Decision boundary 的法向量
            if w[1] != 0:
                x_last_decision_boundary = np.linspace(0,w[1])
                y_last_decision_boundary = (w[2]/w[1])*x_last_decision_boundary
                plt.plot(x_last_decision_boundary, y_last_decision_boundary,'c--')
           
            #(重要)用來更新w
            w += y*x     
        
            print("x: " + str(x))            
            print("w: " + str(w))
            # x向量 
            x_vector = np.linspace(0,x[1])
            y_vector = (x[2]/x[1])*x_vector
            plt.plot(x_vector, y_vector,'b')
            # Decision boundary 的方向向量
            x_decision_boundary = np.linspace(-0.5,7)
            y_decision_boundary = (-w[1]/w[2])*x_decision_boundary - (w[0]/w[2])
            plt.plot(x_decision_boundary, y_decision_boundary,'r')
            # Decision boundary 的法向量(青色的虛線)
            x_decision_boundary_normal_vector = np.linspace(0,w[1])
            y_decision_boundary_normal_vector = (w[2]/w[1])*x_decision_boundary_normal_vector
            plt.plot(x_decision_boundary_normal_vector, y_decision_boundary_normal_vector,'g')
            plt.xlim(-0.5,7.5)
            plt.ylim(5,-3)
            plt.show()

print("\n結論 : ") 
print("最後跑到iterator 9的時候，會找到一個完美的decision boundary把這兩群資料(setosa、versicolor)切開!!")

"""
會把所有的花都判斷成-1類，因此找到第一筆+1類的就會發現預測錯誤，
對w進行更新 w = w+y*x 就是 
[0,0,0]+[1, 5.1,1.4] = [1, 5.1,1.4] 
更新第一輪之後的w

以物理意義來說，代表的是該Decision boundary（紅線）的法向量(綠線)，
在紅線的內以及下方資料點會被歸類成-1類，在紅線的上方會被歸類成+1類

為了方便理解新增了青色的虛線代表上一個Decision boundary的法向量，
藍色線代表發生預測錯誤的資料點，
而 w = w+y*x 而這次的y是-1，
因此 [1, 5.1, 1.4] + -1*[1,7,4.7] = [0, -1.9, -3.3]，就是下方的綠色線

最後跑到iterator 9的時候會找到一個完美的decision boundary把這兩群資料切開
"""


"""
Perception優點：
最簡單的線性分類演算法，Perception演算法的原理可推廣至其他複雜的演算法，
因此許多課程或是書籍皆會以此當作最初的教材。

Perception缺點：
1. 一定要線性可分Perception演算法才會停下來
  （實務上我們沒辦法事先知道資料是否線性可分）
2. Perception演算法的錯誤率不會逐步收斂
3. Perception演算法只知道結果是A類還B類，但沒辦法知道是A, B類的機率是多少（接下來要介紹的Logistic regression可解決此問題）

"""