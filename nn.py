#!/usr/bin/python

# This file is a short script to take in the MINST dataset and train a NN using pybrain

import csv
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import TanhLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
from numpy import loadtxt

#with open( 'train.csv') as csv_file:
#	reader = csv.reader(csv_file)
#	data_set = ClassificationDataSet( 784, 1, nb_classes=10 ) 
#	for row in reader:
#		print row
#		if reader.line_num == 1:
#			break
#		else:
#			data_set.addSample( (row[1:784]), (row[0]) )

inputted_data = loadtxt( 'train.csv', delimiter=',', skiprows=1 )

data_set = ClassificationDataSet( 783, 1, nb_classes=10 )

for i in inputted_data:
	data_set.addSample( i[1:784], i[0] )

test_data, amtrack_data = data_set.splitWithProportion( 0.25 )

amtrack_data._convertToOneOfMany()
test_data._convertToOneOfMany()

print "Number of training patterns: ", len(amtrack_data)
print "Input and output dimensions: ", amtrack_data.indim, amtrack_data.outdim
print "First sample (input, target, class): "
print amtrack_data['input'][0], amtrack_data['target'][0], amtrack_data['class'][0]

net = buildNetwork( amtrack_data.indim, 786, amtrack_data.outdim, hiddenclass=TanhLayer, outclass=SoftmaxLayer )

trainer = BackpropTrainer( net, dataset=amtrack_data, momentum=0.1, verbose=True, weightdecay=0.01)

trainer.trainUntilConvergence()

trainer.testOnData( verbose=True )
