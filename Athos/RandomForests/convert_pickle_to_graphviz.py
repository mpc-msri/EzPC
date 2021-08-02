"""

Authors: Mayank Rathee, Pratik Bhatu.

Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import pickle
from sklearn.tree import export_graphviz

# from sklearn.tree import DecisionTreeClassifier, RandomForestRegressor
from subprocess import call

# import matplotlib.pyplot as plt
import sys
import os


def convert_pickle_to_graphviz(path, task, ml_type, build_dir):
    model_loaded = pickle.load(open(path, "rb"))
    if ml_type == "tree":
        no_of_estim = 1
    else:
        no_of_estim = model_loaded.n_estimators

    print("The specified task is (tree/forest):", ml_type)
    print("This is the number of estimators: ", no_of_estim)

    if task == "reg":
        depth = model_loaded.max_depth
        print("This is the depth: ", depth)

    # with open("decision_tree_stat.txt", "w") as f:
    #     f.write(str(no_of_estim) + "\n")

    print("Exporting model via graphviz...")
    tree_dot_path = os.path.join(build_dir, "tree.dot")
    for i in range(no_of_estim):
        if ml_type == "tree":
            estimator = model_loaded
        else:
            estimator = model_loaded.estimators_[i]
        # Export as dot file
        if task == "cla":
            export_graphviz(
                estimator,
                out_file=tree_dot_path,
                rounded=True,
                proportion=False,
                filled=True,
                class_names=["0", "1"],
            )
        else:
            export_graphviz(
                estimator,
                out_file=tree_dot_path,
                rounded=True,
                proportion=False,
                precision=2,
                filled=True,
            )

        filename = "tree" + str(i) + ".txt"
        filename = os.path.join(build_dir, filename)
        # print("Exported tree #" + filename)
        call(["dot", "-Tplain", tree_dot_path, "-o", filename])
    return no_of_estim
