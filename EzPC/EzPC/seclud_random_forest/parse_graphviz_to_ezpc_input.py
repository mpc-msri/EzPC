'''
This python file takes in a graphviz text file,
creates a tree in memory and outputs the tree's 
characteristic (feature and threshold at each node)
where it is ASSUMED that initially each node of
the tree is either leaf or it has 2 children.
This file also takes care of adding dummy nodes
to create a new funtionally equivalent complete
binary tree to be used by EzPC code.
'''

class TreeNode(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.value = 0
        self.feature = -1
        self.depth = -1

ctr = 0
ezpc_features = []
ezpc_threshold = []
ezpc_depth = []
max_depth = -1

def fill_recur(features, threshold, depth):
    global ctr
    global max_depth
    max_depth = max(max_depth, depth)
    if features[ctr] == -1:
        #Leaf Node
        node = TreeNode()
        node.value = threshold[ctr]
        node.depth = depth
        ctr += 1
        return node
    else:
        node = TreeNode()
        node.value = threshold[ctr]
        node.feature = features[ctr]
        node.depth = depth
        ctr += 1
        node_left = fill_recur(features, threshold, depth+1)
        node_right = fill_recur(features, threshold, depth+1)
        node.left = node_left
        node.right = node_right
        return node

def is_internal(node):
    if node.feature == -1:
        return False
    else:
        return True

def get_to_pad_subtree(node, depth_diff):
    if depth_diff == 1:
        #New leafs
        node_left = TreeNode()
        node_right = TreeNode()
        node_left.value = node.value
        node_right.value = node.value
        node_left.depth = max_depth + 1 - depth_diff
        node_right.depth = max_depth + 1 - depth_diff
        node.left = node_left
        node.right = node_right
        node.feature = 1
        node.value = 0.0
        return node
    else:
        node_left = TreeNode()
        node_right = TreeNode()
        node_left.value = node.value
        node_right.value = node.value
        node_left.feature = node.feature
        node_right.feature = node.feature
        node_left.depth = max_depth + 1 - depth_diff
        node_right.depth = max_depth + 1 - depth_diff
        node_left = get_to_pad_subtree(node_left, depth_diff-1)
        node_right = get_to_pad_subtree(node_right, depth_diff-1)
        node.left = node_left
        node.right = node_right
        node.feature = 1
        node.value = 0.0
        return node
 
def pad_to_complete_tree(node):
    global max_depth
    if not is_internal(node):
        #Leaf node
        if node.depth != max_depth:
            #Needs padding
            node = get_to_pad_subtree(node, max_depth-node.depth)            
    else:
        pad_to_complete_tree(node.left)
        pad_to_complete_tree(node.right)
    

def dump_complete_tree(root):
    global ezpc_features
    global ezpc_threshold
    global ezpc_depth
    global max_depth
    global nodes_in_complete_tree
    queue = [root]
    ctr_local = 0
    while ctr_local < nodes_in_complete_tree:
        current_node = queue[ctr_local]
        ctr_local += 1
        if is_internal(current_node):
            ezpc_features.append(current_node.feature)
            ezpc_threshold.append(current_node.value)
            ezpc_depth.append(current_node.depth)
            queue.append(current_node.left)
            queue.append(current_node.right)
        else:
            ezpc_features.append(-1)
            ezpc_threshold.append(current_node.value)
            ezpc_depth.append(current_node.depth)

import sys

input_file = open(sys.argv[1], 'r')
lines = input_file.readlines()
lines = lines[1:]

depth = 0;
nodes_this_tree = 0

features = []
threshold = []

for i in range(len(lines)):
    curline = lines[i]
    start_location = curline.find("\"") 
    start_location += 1
    if start_location == 0:
        break
    nodes_this_tree += 1
    if curline[start_location] == "X":
        #This is an internal node
        end_location_feature = curline.find("]")
        start_location_th = curline.find("<=")
        end_location_th = curline.find("\\n")
        feature_val = int(curline[start_location+2:end_location_feature])
        threshold_val = float(curline[start_location_th+3:end_location_th])
        features.append(feature_val)
        threshold.append(threshold_val)
        # print("Internal Node")
        # print(feature_val)
        # print(threshold_val)
    else:
        #This is a leaf
        start_location_val = curline.find("value =")
        end_location_val = curline.find("\" filled")
        output_val = float(curline[start_location_val+7:end_location_val])
        features.append(-1)
        threshold.append(output_val)
        # print("Leaf Node")
        # print(output_val)

root = fill_recur(features, threshold, 1)

nodes_in_complete_tree = pow(2, max_depth) - 1
# if nodes_in_complete_tree != nodes_this_tree:
    # print("[PADDING] Input tree not complete. Padding to make complete.")
# else:
    # print("Input tree already complete. No need to pad.")

pad_to_complete_tree(root)

dump_complete_tree(root)

# print(ezpc_features)
# print(ezpc_threshold)
# print(ezpc_depth)
# print(max_depth)

scale_file = open("scaling_factor.txt", 'r')
temp = scale_file.readlines()
scaling_factor = int(temp[0])
scale_file.close()

import math

print("Writing to ezpc_parsed_tree.txt")
print("[FLOAT TO FIXED] Scaling by " + str(scaling_factor) + " times")
output_file = open("ezpc_parsed_tree.txt", 'a')
for i in range(len(ezpc_features)):
    output_file.write(str(ezpc_features[i]) + "\n")
for i in range(len(ezpc_threshold)):
    output_file.write(str(math.floor(scaling_factor*ezpc_threshold[i])) + "\n")

