fl = open("random_forest.ezpc", 'r')
lines = fl.readlines()

# print("Initial params -- before correcting")
# print(lines[5])
# print(lines[6])
# print(lines[7])
# print(lines[8])

fl.close()

param_fl = open("decision_tree_stat.txt", 'r')
new_params = param_fl.readlines()
param_fl.close()

no_of_trees = int(new_params[0])
depth = int(new_params[1])
no_of_nodes = pow(2, depth) - 1
# print("New params are: ")
# print(no_of_trees)
# print(depth)
# print(no_of_nodes)

lines[5] = lines[5][:-1] + str(depth) + ';\n'
lines[6] = lines[6][:-1] + str(depth) + ';\n'
lines[7] = lines[7][:-1] + str(no_of_trees) + ';\n'
lines[8] = lines[8][:-1] + str(no_of_nodes) + ';\n'


new_fl = open("random_forest_main.ezpc", 'w')
for i in range(len(lines)):
    new_fl.write(lines[i])
new_fl.close()

print("Created new ezpc files with correct params: random_forest_main.ezpc")

sf_file = open("scaling_factor.txt", 'r')
temp = sf_file.readlines() 
sf_value = int(temp[0])
sf_file.close()

sklearn_file = open("decision_tree_sklearn_ans.txt", 'r')
temp = sklearn_file.readlines() 
sklearn_file.close()
sklearn_file = open("decision_tree_sklearn_ans.txt", 'a')
skl_ans = float(temp[1])
import math
ezpc_ans = math.floor(skl_ans*sf_value*no_of_trees)
sklearn_file.write("EzPC/ABY/2PC answer should look be: " + str(ezpc_ans) + '\n')

print("To check the correctness of your EzPC/ABY/2PC computation, match the answer with the value in decision_tree_sklearn_ans.txt")
