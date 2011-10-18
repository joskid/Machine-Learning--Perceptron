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
# the vector 
def readin(name):
	reader = csv.reader(open(name, "rU"))
	listValues = []
	
	for row in reader:
		listValues.append(row)
	
	# Because last row is empty
	del listValues[-1]
	
	# Setting the class to 1 or 0
	for row in listValues:
		if row[-1] == "Iris-setosa":
			row[-1] = 1
		else:
			row [-1] = 0
			
	#for row in listValues:
	#	print row
	
	listValuesFloat = [map(float,x) for x in listValues]
	
	return listValuesFloat		



# A helper function that computes the dot product of vectors values and weights
def sumFunction(values, weights):
	return sum(value * weights[index] for index, value in enumerate(values))



# A helper function that comptues the margin of the vectors values and weights
# Given a feature vector x and weight vector w, the margin is: x.w/norm(w)
# I use the absolute value to calculate absolute margin, as trainPerceptron
# already has a way to compute the number of errors
def margin(values,weights):
	wnorm = math.sqrt(sumFunction(weights,weights))
	return abs(sumFunction(values,weights)/wnorm)



# A wrapper function for margin that finds the minimum margin given a set and weights
def findMargin(testSet, weights):
	xLength = (len(testSet[1])-1)
	minMargin = 100.0
	
	for row in testSet:
		inputVector = row[0:xLength]
		thisMargin = margin(inputVector, weights)
		if thisMargin < minMargin:
			minMargin = thisMargin
			
	print "Minimum margin is: ", minMargin



# This runs through the perceptron algorithm and calibrates the weights vector
def trainPerceptron(trainingSet):
	xLength = (len(trainingSet[1])-1)
	threshold = 0.0
	learningRate = 0.1
	weights = [0] * xLength
	passes = 0
	updates = 0
	
	while True:
		#print '-' * 60
		errorCount = 0
		for row in trainingSet:
			inputVector = row[0:xLength]
			desiredOutput = row[-1]
			#print weights
			result = 1 if sumFunction(inputVector, weights) > threshold else 0
			error = desiredOutput - result
			if error != 0:
				errorCount += 1
				for index, value in enumerate(inputVector):
					weights[index] += learningRate * error * value
					updates += 1
		if errorCount == 0 or passes > 1000:
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
			
	print "The percent error is:", (errorCount/testCount)



def main():
	trainSet = readin("irisTrain.txt")
	weights = trainPerceptron(trainSet)
	print "Weights are: \n", weights
	
	findMargin(trainSet,weights)
	
	testSet = readin("irisTest.txt")
	testPerceptron(testSet,weights)
	
	#x = [1,2,3,4,5]
	#y = [5,4,3,2,1]
	#print margin(x,y)
		
	#x = [5.6,2.7,4.2,1.3]
	#print sumFunction(x,weights)


if __name__ == '__main__':
	main()

