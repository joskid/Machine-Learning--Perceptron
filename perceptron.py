# Patrick Grennan
# grennan@nyu.edu
#
# Machine Learning
# Homework 2: Perceptron
# This is a collection of interacting scripts that run the perceptron algorithm
# on a given (linearly separable) data set and outputs a linear classifier
# in the form a a weight vector. (To classify a given x you have to take the 
# sign of the dot product of x and the weight vector)

import sys
import os
import csv
import math

# A helper function that reads in a comma seperated file and outputs a list
# The last value in the outputed matrix is the correct classification for
# the vector, so when it is read it 
def readin(name):
	reader = csv.reader(open(name, "rU"))
	listValues = []
	
	for row in reader:
		listValues.append(row)
	
	# Because last row is empty in the data sets
	del listValues[-1]
	
	# Setting the class to 1 or 0 (**not needed for spam class**)
	for row in listValues:
		if row[-1] == "R":
			row[-1] = 1
		else:
			row [-1] = 0
			
	listValuesFloat = [map(float,x) for x in listValues]
	
	return listValuesFloat		



# A helper function that computes the dot product of vectors values and weights
def sumFunction(values, weights):
	return sum(value * weights[index] for index, value in enumerate(values))



# A helper function that comptues the margin of the vectors values and weights
# Given a feature vector x and weight vector w, the margin is: x.w/norm(w)
# I use the absolute value to calculate absolute margin, as trainPerceptron
# already has a way to compute the number of errors
def margin(values,weights, desiredOutput):
	if desiredOutput == 0:
		desiredOutput = -1
	wnorm = math.sqrt(sumFunction(weights,weights))
	return desiredOutput*sumFunction(values,weights)/wnorm



# A wrapper function for margin that finds the minimum margin given a set and weights
def findMargin(testSet, weights):
	xLength = (len(testSet[1])-1)
	minMargin = 100.0
	
	for row in testSet:
		inputVector = row[0:xLength]
		desiredOutput = row[-1]
		thisMargin = abs(margin(inputVector, weights,desiredOutput))
		if thisMargin < minMargin:
			minMargin = thisMargin
			
	print "Minimum margin is: ", minMargin
	return minMargin



# This runs through the perceptron algorithm and calibrates the weights vector
def trainPerceptron(trainingSet):
	xLength = (len(trainingSet[1])-1)
	weights = [0] * xLength
	passes = 0
	updates = 0
	
	while True:
		errorCount = 0
		for row in trainingSet:
			inputVector = row[0:xLength]
			desiredOutput = row[-1]
			result = 1 if sumFunction(inputVector, weights) > 0 else 0
			error = desiredOutput - result
			if error != 0:
				errorCount += 1
				for index, value in enumerate(inputVector):
					weights[index] += error * value
				updates += 1
		if errorCount == 0 or passes > 1000:
			break
		else:
			passes += 1
			
	print "Number of passes:" , passes
	print "Number of updates:" , updates
	
	return weights



# This runs through the perceptron algorithm and calibrates the weights vector
# by checking to see if it is within the threshold given by the formula given
# by < p(1-e) with p being the optimal margin (the old margin is being used)
# and e being an integer in the range (0,1). This update guarantees that it
# will converge in O((R^2/p^2)/(1-e)) updates, with R being the radius of the
# sphere containing the sample points
def modifiedPerceptron(trainingSet, givenMargin):
	xLength = (len(trainingSet[1])-1)
	weights = [0] * xLength
	passes = 0
	updates = 0
	marginTest = givenMargin*(1 - 0.25)
	
	while True:
		errorCount = 0
		for row in trainingSet:
			inputVector = row[0:xLength]
			desiredOutput = row[-1]
			
			# This is only checking the margin of one vector
			if weights[0] == 0.0 or (margin(inputVector, weights, desiredOutput) < marginTest):
				errorCount += 1
				for index, value in enumerate(inputVector):
					if desiredOutput == 0:
						desiredOutput = -1
					weights[index] += desiredOutput * value
				updates += 1
				
		if errorCount == 0 or passes > 500:
			break
		else:
			passes += 1
			
	print "Number of passes:" , passes
	print "Number of updates:" , updates
		
	return weights
		



# This tests the accuracy of the perceptron on the test set
def testPerceptron(testSet, weights):
	xLength = (len(testSet[1])-1)
	threshold = 0.0
	errorCount = 0.0
	testCount = 0.0
	
	for row in testSet:
		inputVector = row[0:xLength]
		desiredOutput = row[-1]
		result = 1 if sumFunction(inputVector, weights) > threshold else 0
		#print result
		error = desiredOutput - result
		if error != 0:
			errorCount += 1
		testCount += 1
			
	print "The percent error is:", (errorCount/testCount)*100



def main():
	trainSet = readin("sonarTrain.txt")
	weights = trainPerceptron(trainSet)
	print "Weights are: \n", weights
	
	margin = findMargin(trainSet,weights)
	
	testSet = readin("sonarTest.txt")
	testPerceptron(testSet,weights)
	
	print "\n\n"
	modWeights = modifiedPerceptron(trainSet,margin)
	findMargin(trainSet,modWeights)
	testPerceptron(testSet,modWeights)
	


if __name__ == '__main__':
	main()

