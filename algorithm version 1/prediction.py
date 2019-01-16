import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
# %matplotlib inline

train_data = None 		# pandas data structure used to train our program
lineParameters = None 	# a dictionary of all line parameters (from getLineParameters(variable)) for each variable
correlations = None 	# a dictionary of correlations of each variable to salePrice
test_data = None 		# pandas data structure used to test our program


def getLineParameters(varName):
	# returns (a,b) of a line y=ax+b of best fit for the variable provided
	# uses linear regression to get a and b
	return tuple(np.polyfit(train_data[varName], train_data['SalePrice'], 1))
	
def calculateCorrelations():
	global correlations
	correlations = train_data.corr()

def getCorr(varName):
	# return the correlation between the variable and sale price as a number
	return correlations['SalePrice'][varName]

def loadData(trainFile, testFile):
	# writes the train data from a file to train_data
	global train_data
	train_data = pd.read_csv(trainFile)
	train_data.fillna(0, inplace=True)
	
	global test_data
	test_data = pd.read_csv(testFile)#train_data[len(train_data)//2:]
	test_data.fillna(0, inplace=True)
	
	for varName in train_data:
		
		type = train_data.dtypes[varName]
		if(type not in ["int64","float64"]):
			# the variable is categorical, not numerical
			
			categories = train_data[varName].unique()
			means = train_data.groupby(varName).mean()['SalePrice']
			# print(train_data[varName])
			for category in categories:
				# print(category, means[category])
				
				train_data.loc[train_data[varName]==category,varName] = means[category]
				test_data.loc[test_data[varName]==category,varName] = means[category]
				
				
			for category in test_data[varName].unique():
				if(category not in categories):
					test_data.loc[test_data[varName]==category,varName] = 0
				
			train_data[varName] = train_data[varName].astype('float64', copy=False)
			test_data[varName] = test_data[varName].astype('float64', copy=False)
			# print(train_data[varName])
	
	calculateCorrelations()

def getEstimate(varName, value):
	a, b = getLineParameters(varName)
	estimatedSalePrice = a*value+b
	return estimatedSalePrice
	
def getEstimatedValues(varName):
	a, b = getLineParameters(varName)
	return [a*x+b for x in train_data[varName]]
	
def getPredictedPrice(variables, treshold):
	# variables: pandas row of all variables 
	#            provided to the program {varName: varValue}

	# for each variable we estimate the 
	# salePrice (using getEstimate())

	# finally, calculate a weighted average where values are the 
	# estimated salePrices and weights are correlations of the variables
	
	# the estimated salePrice is that average, which is returned
	#========================================
	
	#a dictionary of correlations of variables and the price predicted with that variable
	
	estimates = {} #{varName : (estimatedPrice, correlationToSalePrice)}
	
	for varName in train_data.columns[:-1]:
		try:
			varVal = variables[varName]
			estimate = getEstimate(varName, varVal)
			corr = getCorr(varName)
			
			if(corr > treshold):
				estimates[varName] = (estimate, corr)
				# print(varName, (estimate, corr))
		except:
			# print(varName, "error")
			pass
		
	corrSum = 0
	estimatedPrice = 0
	
	for varName, (estimate, corr) in estimates.items():
		corrSum += corr
		estimatedPrice += estimate
	if(corrSum == 0):
		return 0
	return estimatedPrice/corrSum

def showScatterPlotOf(variable):
	data = pd.concat([train_data['SalePrice'], train_data[variable]], axis=1)
	a,b = getLineParameters(variable)
	print(a,b)
	plt.plot(train_data[variable], train_data["SalePrice"], 'b.')
	plt.plot(train_data[variable], getEstimatedValues(variable), 'r-')
	plt.show()
	
def showBoxPlotOf(variable):
	data = pd.concat([train_data['SalePrice'], train_data[variable]], axis=1)
	f, ax = plt.subplots(figsize=(8, 6))
	fig = sns.boxplot(x=variable, y="SalePrice", data=data)
	fig.axis(ymin=0, ymax=800000)
	plt.show()

loadData("../data/train.csv", "../data/test.csv")

treshold = 0.75

for index, row in test_data.iterrows():
	predicted = getPredictedPrice(row, treshold)
	id = row['Id']
	print(str(id)+", "+str(predicted))


