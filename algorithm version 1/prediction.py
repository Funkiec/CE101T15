import pandas

train_data = None 		# pandas data structure used to train our program
lineParameters = None 	# a dictionary of all line parameters (from getLineParameters(variable)) for each variable
correlations = None 	# a dictionary of correlations of each variable to salePrice
test_data = None 		# pandas data structure used to test our program


def getLineParameters(varName):
	# returns (a,b) of a line y=ax+b of best fit for the variable provided
	# uses linear regression to get a and b

def getCorr(varName):
	# return the correlation between the variable and sale price as a number

def loadData(filename)
	# writes the train data from a file to train_data

def getEstimate(varName, value)
	# a, b= getLineParameters(variable)
	# estimatedSalePrice = a*value+b
	# return estimatedSalePrice
	
def getPredictedPrice(variables)
	# variables: pandas row of all variables 
	#            provided to the program {varName: varValue}

	# for each variable we estimate the 
	# salePrice (using getEstimate())

	# finally, calculate a weighted average where values are the 
	# estimated salePrices and weights are correlations of the variables
	
	# the estimated salePrice is that average, which is returned
