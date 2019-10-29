import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.tree import export_graphviz
from subprocess import call
import sys

name = sys.argv[1]

print("Loading pickle model...")
model_loaded = pickle.load(open(name, 'rb'))
print("Pickle model loaded")

depth = model_loaded.max_depth
no_of_estim = model_loaded.n_estimators

from sklearn.tree import export_graphviz
from subprocess import call
fl = open("decision_tree_stat.txt", 'w')
fl.write(str(no_of_estim) + '\n')
fl.write(str(depth+1))
fl.close()
print("Exporting model via graphviz...")
for i in range(no_of_estim):
    estimator = model_loaded.estimators_[i]
    # Export as dot file
    export_graphviz(estimator, out_file='tree.dot',
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)
    filename = "tree" + str(i) + ".txt"
    print("Exported tree #" + filename)
    call(['dot', '-Tplain', 'tree.dot', '-o', filename])
