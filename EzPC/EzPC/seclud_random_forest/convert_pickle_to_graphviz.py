
import numpy as np
import pandas as pd
import pickle
from sklearn.tree import export_graphviz
from subprocess import call
import matplotlib.pyplot as plt
import sys

name = sys.argv[1]
task = sys.argv[2]
ml_type = sys.argv[3]

print("Loading pickle model...")
model_loaded = pickle.load(open(name, 'rb'))
print("Pickle model loaded")

if(ml_type == 'tree'):
    no_of_estim = 1
    from sklearn.tree import DecisionTreeClassifier
else:
    no_of_estim = model_loaded.n_estimators
    from sklearn.ensemble import RandomForestRegressor

print("The specified task is (tree/forest):", ml_type)
print("This is the number of estimators: ", no_of_estim)

if(task == 'reg'):
    depth = model_loaded.max_depth
    print("This is the depth: ", depth)

from sklearn.tree import export_graphviz
from subprocess import call
fl = open("decision_tree_stat.txt", 'w')
fl.write(str(no_of_estim) + '\n')
#if(task == 'reg'):
    #fl.write(str(depth+1))
fl.close()
print("Exporting model via graphviz...")
for i in range(no_of_estim):
    if(ml_type == 'tree'):
        estimator = model_loaded
    else:
        estimator = model_loaded.estimators_[i]
    # Export as dot file
    if(task == 'cla'):
        export_graphviz(estimator, out_file='tree.dot',
                    rounded = True, proportion = False, 
                    filled = True, class_names = ['0', '1'])
    else:
        export_graphviz(estimator, out_file='tree.dot',
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)

    filename = "tree" + str(i) + ".txt"
    print("Exported tree #" + filename)
    call(['dot', '-Tplain', 'tree.dot', '-o', filename])
