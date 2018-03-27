import os
import numpy as np

def loadDataset():
	# 处理训练数据集
	trainDataset = []
	trainLabel = []
	trainParentPath = "/Users/Fei/Documents/work/ML/data/KNN/trainingDigits" # 训练数据集位置
	trainParentList = os.listdir(trainParentPath)
	# 循环训练数据集中的每一个文件，文件名是有类名_序号构成
	for trainFile in trainParentList:
		trainLabel.append(trainFile.split("_")[0]) #分类，就是该文件代表的数字
		with open(trainParentPath+"/"+trainFile) as fopen:
			dataset = []
			for lines in fopen.readlines():
				dataset.append([ int(x) for x in lines.strip()])


			trainDataset.append(dataset)
	# 处理测试数据集
	testDataset = []
	testLabel = []
	testParentPath = "/Users/Fei/Documents/work/ML/data/KNN/testDigits" # 测试训练集位置
	testParentList = os.listdir(testParentPath)
	for testFile in testParentList:
		testLabel.append(testFile.split("_")[0])
		with open(testParentPath+"/"+testFile) as fopen:
			dataset = []
			for lines in fopen.readlines():
				dataset.append([int(x) for x in lines.strip()])
			testDataset.append(dataset)
	return trainDataset, trainLabel, testDataset, testLabel


def distance(datasetX, datasetY):
	return np.sqrt(np.sum(np.power(np.array(datasetX).ravel() - np.array(datasetY).ravel(),2)))

if __name__ == '__main__':
	k = 20
	trainDataset, trainLabel, testDataset, testLabel = loadDataset()
	errCount = 0
	for index, tempTestData in enumerate(testDataset): #循环每一个测试集的数据
		trainDistance = []
		for tempTrainData in trainDataset: #循环每一个训练集的数据，计算与之的距离
			trainDistance.append(distance(tempTrainData,tempTestData))
		# 排序选出其中最小的k个元素的位置
		maxTrainDistanceIndex = np.array(trainDistance).argsort()
		# 最大的k个元素的分类结果
		maxTrainLabel = np.array(trainLabel)[maxTrainDistanceIndex[:k]]
		# 预测的类
		kind = maxTrainLabel[maxTrainLabel.argmax()]
		if kind != testLabel[index]:
			errCount += 1
			print("应该是:"+testLabel[index]+"预测为:"+kind)
	print("错误率为："+str(errCount/len(testLabel)))