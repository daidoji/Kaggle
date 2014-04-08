import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn import neighbors, cross_validation

n_neighbors = 10

data_input = np.loadtxt( 'train.csv', delimiter=',', skiprows=1 )
print "Loaded data"

print "sampling"
X_train, X_test, y_train, y_test = cross_validation.train_test_split( 
	data_input[ :, 1:784 ], data_input[ :, 0 ], test_size = 0.25 )

print "fitting"
clf = neighbors.KNeighborsClassifier( n_neighbors = 10, weights = 'uniform', algorithm = 'kd_tree')
clf.fit( X_train, y_train )

print "scoring"
print clf.score( X_test, y_test )
