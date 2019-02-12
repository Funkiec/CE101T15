import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import warnings
# %matplotlib inline

warnings.simplefilter('ignore', np.RankWarning)

train_data = None 		# pandas data structure used to train our program
lineParameters = None 	# a dictionary of all line parameters (from getLineParameters(variable)) for each variable
correlations = None 	# a dictionary of correlations of each variable to salePrice
test_data = None 		# pandas data structure used to test our program
mseValues = None        # dict of mse values 
varBoundaries = None	# dict of tuples containing the min and max values of each variable in train_data


polyDeg = 5 #degree of the polynomial fitted to each variable

def getFunctionParameters(varName):
	# returns (a,b) of a line y=ax+b of best fit for the variable provided
	# uses linear regression to get a and b
	return tuple(np.polyfit(train_data[varName], train_data['SalePrice'], polyDeg))
	
def calculateCorrelations():
	global correlations
	correlations = train_data.corr()

def getCorr(varName):
	# return the correlation between the variable and sale price as a number
	return correlations['SalePrice'][varName]
	
def getMSE(varName):
	return getEstimatedValues(varName)[1]

def storeMSE():
	global mseValues
	mseValues = {}
	for varName in train_data.columns[:-1]:
		mseValue = getMSE(varName)
		mseValues[varName] = mseValue

def loadData(trainFile, testFile):
	# writes the train data from a file to train_data
	global train_data
	train_data = pd.read_csv(trainFile)
	
	#how much data is missing from each variable
	missingPercentage = train_data.isnull().sum()/train_data.isnull().count()
	#removing each variable which is missing in at least 10% of cases
	toRemove = []
	for column in train_data.columns:
		if(missingPercentage[column] >= 0.1):	
			print("Removing",column)
			toRemove.append(column)
	
	for column in toRemove:
		del train_data[column]
		
	global varBoundaries 
	varBoundaries = {}
	
	train_data.fillna(0, inplace=True)
	
	foundBoundaries = []
	for varName in train_data.columns:
		try:
			type = train_data.dtypes[varName]
			if(type in ["int64","float64"]):
				minimum = train_data[train_data[varName] != np.nan][varName].min()
				maximum = train_data[train_data[varName] != np.nan][varName].max()
				varBoundaries[varName] = (minimum, maximum)
				foundBoundaries.append(varName)
		except:
			pass
	
	global test_data
	test_data = pd.read_csv(testFile)#train_data[len(train_data)//2:]
	for column in toRemove:
		del test_data[column]
	test_data.fillna(0, inplace=True)
	
	for varName in train_data:
		
		type = train_data.dtypes[varName]
		# print(varName, "(", type, "):")
		if(type not in ["int64","float64"]):
			# the variable is categorical, not numerical
			
			categories = train_data[varName].unique()
			means = train_data.groupby(varName).mean()['SalePrice']
			# print(train_data[varName])
			for category in categories:
				# print(category, means[category])
				
				train_data.loc[train_data[varName]==category,varName] = means[category]
				test_data.loc[test_data[varName]==category,varName] = means[category]
				# print("\t", "{0:10}".format(str(category)), "==>\t", means[category])
				
				
			for category in test_data[varName].unique():
				if(category not in categories):
					test_data.loc[test_data[varName]==category,varName] = -1
				
			train_data[varName] = train_data[varName].astype('float64', copy=False)
			test_data[varName] = test_data[varName].astype('float64', copy=False)
			# print(train_data[varName])
		# else:
			# print("\t", "<{:<10};{:>10}>".format(varBoundaries[varName][0], varBoundaries[varName][1]))
		if(varName not in foundBoundaries):
			minimum = train_data[train_data[varName] != np.nan][varName].min()
			maximum = train_data[train_data[varName] != np.nan][varName].max()
			varBoundaries[varName] = (minimum, maximum)
			foundBoundaries.append(varName)
			
	
	calculateCorrelations()
	storeMSE()

def calcPolynomial(x, poly):
	result = 0
	for power, coefficient in enumerate(poly):
		result += coefficient * x**(polyDeg-power)
	return result
	
def getEstimate(varName, value):
	poly = getFunctionParameters(varName)
	estimatedSalePrice = calcPolynomial(value, poly)
	return estimatedSalePrice
	
def getEstimatedValues(varName):
	poly = getFunctionParameters(varName)
	values = [calcPolynomial(x, poly) for x in train_data[varName]]
	mse = 0
	for index, x in enumerate(train_data[varName]):
		actual_y = train_data["SalePrice"][index]
		estimate_y = values[index]
		mse += (actual_y - estimate_y)**2
	return values, mse/len(values)
	
def getPredictedPrice(variables, variablesToUse):
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
	
	filtered = []
	
	for varName in train_data.columns[:-1]:
		varVal = variables[varName]

		#if the value is in the range of values as in the train_data
		# print('min', varBoundaries[varName][0])
		# print('val', varVal, type(varVal))
		# print('max', varBoundaries[varName][1])
		power10 = 10**10
		if(varBoundaries[varName][0] <= varVal <= varBoundaries[varName][1]):
			weight = power10/mseValues[varName]
			filtered.append((varName, weight))
		else:
			# print("NOT using", varName, "=", varVal)
			pass
			
	filtered.sort(key = lambda v: -v[1])
	# print()
	weightPower = 3
	for varName, weight in filtered[:variablesToUse]:
		try:
			varVal = variables[varName]
			estimate = getEstimate(varName, varVal)			
			estimates[varName] = (estimate, weight**weightPower)
			# print(varName, (estimate, weight))
		except:
			# print(varName, "error")
			pass
		
	weightSum = 0
	estimatedPrice = 0
	
	for varName, (estimate, weight) in estimates.items():
		weightSum += weight
		estimatedPrice += estimate*weight
	if(weightSum == 0):
		return 0
	return estimatedPrice/weightSum

def scatterPlotOf(variable):
	data = pd.concat([train_data['SalePrice'], train_data[variable]], axis=1)
	estimates, r2 = getEstimatedValues(variable)
	# print(a,b)
	plt.plot(train_data[variable], train_data["SalePrice"], 'b.')
	plt.plot(train_data[variable], estimates, 'r.')
	plt.xlabel(variable)

	
def showBoxPlotOf(variable):
	data = pd.concat([train_data['SalePrice'], train_data[variable]], axis=1)
	f, ax = plt.subplots(figsize=(8, 6))
	fig = sns.boxplot(x=variable, y="SalePrice", data=data)
	fig.axis(ymin=0, ymax=800000)
	plt.show()

loadData("../data/train.csv", "../data/test.csv")

# for varName, (min, max) in varBoundaries.items():
	# print(varName.format("s20"), min, max)

predictedData = {"Id":[],"SalePrice":[]}
for index, row in list(test_data.iterrows()):
	
	#								   ||
	#predict the price based on 	   \/	best variables
	predicted = getPredictedPrice(row, 5)
	# print("\nPredicted:", predicted)
	
	#saving the result to a .csv file
	id = row['Id']
	predictedData["Id"].append(int(id))
	predictedData["SalePrice"].append(predicted)
	# print(str(id)+", "+str(predicted))
	
pd.DataFrame(predictedData).to_csv("predicted.csv", columns=["Id","SalePrice"], index=False)



