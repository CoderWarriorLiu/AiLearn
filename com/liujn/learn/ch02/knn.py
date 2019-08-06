###KNN

from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

average()

#分类计算
def classify0(inx, dataSet, labels, k):
    """计算k近邻"""
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inx, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        vateILabel = labels[sortedDistIndicies[i]]
        classCount[vateILabel] = classCount.get(vateILabel,0) + 1

    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #返回类型
    return sortedClassCount[0][0]


