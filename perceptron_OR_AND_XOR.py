#
from perceptron import Perceptron
import numpy as np
 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#construct the OR dataset
y_or = np.array([[0], [1], [1], [1]])
#construct the AND dataset
y_and = np.array([[0], [0], [0], [1]])
#construct the XOR dataset
y_xor = np.array([[1], [0], [0], [1]])

print("X.shape : ", X.shape)
 
#define perceptron and train it---------
#perceptron用於新建一個兩層神經網路
print("[1]--------Training perceptron--------")
#x.shape:是x中每個樣品的參數個數/ alpha:梯度下降,更新權值的速度
p = Perceptron(X.shape[1], alpha=0.1) 
#fit(樣品的輸入參數矩陣, 樣品輸入參數的輸出矩陣, 迭代次數)
p.fit(X, y_or, epochs=20)
 
#now that perceptron is trained we can eval it.
print("[1]--------Testing perceptron OR--------")

#now that network is trained,loop over the data points.
for (x, target) in zip(X, y_or):
    #make a prediction on the data points & display the results to our console
	pred = p.predict(x)
    #data:樣品輸入參數/ ground-truth(truth):真實值/ pred:預測結果 
	print("[1] data={}, truth={}, pred={}".format(x, target[0], pred))
 
    
print("[2]--------Training perceptron AND--------")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y_and, epochs=20)
 
print("[2]--------Testing perceptron AND--------")
for (x, target) in zip(X, y_and):
	pred = p.predict(x)
	print("[2] data={}, truth={}, pred={}".format(x, target[0], pred))
 
print("[3]--------Training perceptron XOR--------")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y_xor, epochs=200)
 
print("[3]--------Testing perceptron XOR--------")
for (x, target) in zip(X, y_xor):
	pred = p.predict(x)
	print("[3] data={}, truth={}, pred={}".format(x, target[0], pred))


print("\n結論 : ") 
print("OR 和 AND 可用一條直線將0 1樣本分類-兩層神經網路就可實現預測結果。\n")
print("XOR 之預測結果不準確，因為只具備兩層神經元而沒有隱藏層之神經網路，無法對非線性樣本進行分類!\n")
