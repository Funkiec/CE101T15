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
test_data = None 		# pandas data structure used to test our program
mseValues = None        # dict of mse values 
varBoundaries = None	# dict of tuples containing the min and max values of each variable in train_data
functions = None		# dict of functions (polynomials of degree 'polyDeg') of best fit for each variable


polyDeg = 5 #degree of the polynomial fitted to each variable

def getFunctionParameters(varName):
	# returns coefficients=(a_1, a_2, ..., a_polyDeg) of a polynomial 
	# y = (a_1)x^polyDeg + (a_2)x^(polydeg-1) + ... (a_polyDeg)x 
	# of best fit for the variable provided
	
	# uses multiple linear regression to get the coefficients
	return tuple(np.polyfit(train_data[varName], train_data['SalePrice'], polyDeg))
	
def getMSE(varName):
	# returns the mean squared error of the estimated function SalePrice(varName)
	global mseValues
	if(mseValues == None):
		storeMSE()
	return mseValues[varName]

def storeMSE():
	# calculates and saves all the mean squared error values to a dictionary
	global mseValues
	mseValues = {}
	for varName in train_data.columns[:-1]:
		mseValue = getEstimatedValues(varName)[1]
		mseValues[varName] = mseValue

def loadData(trainFile, testFile):
	# reads the train and test data from a file to train_data and test_data
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
		
		
	# saving the proper range in which each variable takes value
	#this will be used to check if a value is a proper and useful value
	global varBoundaries 
	varBoundaries = {}	
	
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
			
	# replacing all missing data with 0s (this won't be an issue, since
	#the value will be later checked using 'varBoundaries'
	train_data.fillna(0, inplace=True)
	
	# loading test data
	global test_data
	test_data = pd.read_csv(testFile)
	
	# removing the same columns which were removed from train data
	for column in toRemove:
		del test_data[column]
	
	# replacing missing data with 0s as with the train data
	test_data.fillna(0, inplace=True)
	
	# assigning numerical values to categorical data
	for varName in train_data:
		type = train_data.dtypes[varName]
		
		if(type not in ["int64","float64"]):
			# the variable is categorical, not numerical
			
			categories = train_data[varName].unique()
			means = train_data.groupby(varName).mean()['SalePrice']
			
			# each category value is replaced by the mean value of SalePrice from rows
			#in which this value occured
			for category in categories:				
				train_data.loc[train_data[varName]==category,varName] = means[category]
				test_data.loc[test_data[varName]==category,varName] = means[category]				
				
			# this is just in case there are any categories in test data
			#which didn't occur in train data
			for category in test_data[varName].unique():
				if(category not in categories):
					test_data.loc[test_data[varName]==category,varName] = -1
					# this -1 value won't be a problem, sice 'varBoundaries' will be used
				
			# saving the new, replaced values
			train_data[varName] = train_data[varName].astype('float64', copy=False)
			test_data[varName]  = test_data[varName].astype('float64', copy=False)

		if(varName not in foundBoundaries):
			# updating 'varBoundaries' with those new numerical values
			minimum = train_data[train_data[varName] != np.nan][varName].min()
			maximum = train_data[train_data[varName] != np.nan][varName].max()
			varBoundaries[varName] = (minimum, maximum)
			foundBoundaries.append(varName)
			
def train():	
	# filling the 'functions' dictionary
	global functions
	functions = {}
	for varName in train_data.columns[:-1]:
		functions[varName] = getFunctionParameters(varName)
	
	# saving the mean squared errors for those functions
	storeMSE()

def calcPolynomial(x, poly):
	# returns the value f(x), where f is the polynomial 'poly'
	result = 0
	for power, coefficient in enumerate(poly):
		result += coefficient * x**(polyDeg-power)
	return result
	
def getEstimate(varName, value):
	# returns and estimate of SalePrice based just on a single variable - 'varName'
	poly = functions[varName]
	estimatedSalePrice = calcPolynomial(value, poly)
	return estimatedSalePrice
	
def getEstimatedValues(varName):
	poly = functions[varName]
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

	# for each variable an estimate of the 
	# salePrice is calculated (using getEstimate())

	# finally, calculate a weighted average where values are the 
	# estimated salePrices and weights are inverses of MSE of the variables
	
	# the estimated salePrice is that average, which is returned
	#========================================
	
	#a dictionary of MSE of variables and the price predicted with that variable
	estimates = {} #{varName : (estimatedPrice, MseToSalePrice)}
	
	#a list of variables to use
	filtered = []
	
	#choosing which variables to use
	for varName in train_data.columns[:-1]:
		varVal = variables[varName]
		
		power10 = 10**10	#This is to make the number closer to 0 (the inverse of MSE is very small).
							#It shouldn't affect the result, but the number is more readable for debugging
		
		if(varBoundaries[varName][0] <= varVal <= varBoundaries[varName][1]):
			#the value is in the range of values as in the train_data
			#therefore the variable is trustworthy
			weight = power10/getMSE(varName)
			filtered.append((varName, weight))
		else:
			#NOT using variable (varName), because the value is
			#outside the allowed range
			pass
			
	#sorting them by their importance (from the most important to the least important)
	filtered.sort(key = lambda v: -v[1])
	
	weightPower = 3	#the weight will be raised to that power
	for varName, weight in filtered[:variablesToUse]:
		try:
			varVal = variables[varName]
			estimate = getEstimate(varName, varVal)		
			
			#the 'weight' is raised to the power of 'weightPower'
			#so the weights have a stronger effect and are 'further apart'
			#from each other
			estimates[varName] = (estimate, weight**weightPower)
		except:
			#this is just in case
			pass
		
		
	#calculating the final SalePrice - the weighted average	
	weightSum = 0
	estimatedPrice = 0
	
	for varName, (estimate, weight) in estimates.items():
		weightSum += weight
		estimatedPrice += estimate*weight
	if(weightSum == 0):	
		#This happens when there weren't any useful variables
		return 0
		
	#resturn the final estimate
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
train()

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



