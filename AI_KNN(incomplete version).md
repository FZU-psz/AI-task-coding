因为之前学过(抄过)一个用knn的算法实现对手写数字的分类的分类器 ，所以对于MNIST想要自己尝试一下用knn做一下看下精确度，咳咳不过时间受限，后面有时间就立马尝试，主要用numpy 也没有用d2l啥的，都是自定义的函数
给出一个文件实例，这个是 0_0.txt文件
00000000000001111000000000000000
00000000000011111110000000000000
00000000001111111111000000000000
00000001111111111111100000000000
00000001111111011111100000000000
00000011111110000011110000000000
00000011111110000000111000000000
00000011111110000000111100000000
00000011111110000000011100000000
00000011111110000000011100000000
00000011111100000000011110000000
00000011111100000000001110000000
00000011111100000000001110000000
00000001111110000000000111000000
00000001111110000000000111000000
00000001111110000000000111000000
00000001111110000000000111000000
00000011111110000000001111000000
00000011110110000000001111000000
00000011110000000000011110000000
00000001111000000000001111000000
00000001111000000000011111000000
00000001111000000000111110000000
00000001111000000001111100000000
00000000111000000111111000000000
00000000111100011111110000000000
00000000111111111111110000000000
00000000011111111111110000000000
00000000011111111111100000000000
00000000001111111110000000000000
00000000000111110000000000000000
00000000000011000000000000000000


```python
from numpy import *
import operator
def img2vector(filename): #将二维图片转换成向量
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j  in range(32):
            returnVect [0,32*i+j] = int(lineStr[j])
    return returnVect
#测试一下
```


```python
filename = "D:\MachineLearning\machinelearninginaction\Ch02\\testDigits\\0_0.txt"

def classify0(inX, dataSet ,labels ,k):
    dataSetSize = dataSet.shape[0]
    diffMat =  tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDiffMat =sqDiffMat.sum(axis=1)
    distances = sqDiffMat**0.5
    sortedDistIndicies =  distances.argsort()
    classCount ={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedclassCount = sorted (classCount.items() , key =operator.itemgetter(1),reverse=True)
    return sortedclassCount[0][0]
```


```python
from os import listdir
def handwritingClassfier():
    hwLabels =[]
    trainingFlieList = listdir("D:\MachineLearning\machinelearninginaction\Ch02\\trainingDigits")
    m = len (trainingFlieList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFlieList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector("D:\MachineLearning\machinelearninginaction\Ch02\\trainingDigits"+"/"+fileNameStr)
    testFileList = listdir ("D:\MachineLearning\machinelearninginaction\Ch02\\testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr =fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("D:\MachineLearning\machinelearninginaction\Ch02\\testDigits"+'/'+fileNameStr)
        classfierResult = classify0(vectorUnderTest, trainingMat,hwLabels,3)
        if classfierResult != classNumStr: errorCount +=1.0
    print("the total error rate is :{:.3f}".format(errorCount/float(mTest)))
```


```python
handwritingClassfier()
```
