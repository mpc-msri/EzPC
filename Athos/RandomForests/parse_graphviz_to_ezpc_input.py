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
"""
This python file takes in a graphviz text file,
creates a tree in memory and outputs the tree's 
characteristic (feature and threshold at each node)
where it is ASSUMED that initially each node of
the tree is either leaf or it has 2 children.
This file also takes care of adding dummy nodes
to create a new funtionally equivalent complete
binary tree to be used by EzPC code.
"""

import math
import os


class TreeNode(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.value = 0
        self.feature = -1
        self.depth = -1


def fill_recur(ctx, features, threshold, depth):
    ctx.max_depth = max(ctx.max_depth, depth)
    if features[ctx.ctr] == -1:
        # Leaf Node
        node = TreeNode()
        node.value = threshold[ctx.ctr]
        node.depth = depth
        ctx.ctr += 1
        return node
    else:
        node = TreeNode()
        node.value = threshold[ctx.ctr]
        node.feature = features[ctx.ctr]
        node.depth = depth
        ctx.ctr += 1
        node_left = fill_recur(ctx, features, threshold, depth + 1)
        node_right = fill_recur(ctx, features, threshold, depth + 1)
        node.left = node_left
        node.right = node_right
        return node


def is_internal(node):
    if node.feature == -1:
        return False
    else:
        return True


def get_to_pad_subtree(ctx, node, depth_diff):
    if depth_diff == 1:
        # New leafs
        node_left = TreeNode()
        node_right = TreeNode()
        node_left.value = node.value
        node_right.value = node.value
        node_left.depth = ctx.max_depth + 1 - depth_diff
        node_right.depth = ctx.max_depth + 1 - depth_diff
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
        node_left.depth = ctx.max_depth + 1 - depth_diff
        node_right.depth = ctx.max_depth + 1 - depth_diff
        node_left = get_to_pad_subtree(ctx, node_left, depth_diff - 1)
        node_right = get_to_pad_subtree(ctx, node_right, depth_diff - 1)
        node.left = node_left
        node.right = node_right
        node.feature = 1
        node.value = 0.0
        return node


def pad_to_complete_tree(ctx, node):
    if not is_internal(node):
        # Leaf node
        if node.depth != ctx.max_depth:
            # Needs padding
            node = get_to_pad_subtree(ctx, node, ctx.max_depth - node.depth)
    else:
        pad_to_complete_tree(ctx, node.left)
        pad_to_complete_tree(ctx, node.right)


def dump_complete_tree(ctx, root):
    queue = [root]
    ctr_local = 0
    while ctr_local < ctx.nodes_in_complete_tree:
        current_node = queue[ctr_local]
        ctr_local += 1
        if is_internal(current_node):
            ctx.ezpc_features.append(current_node.feature)
            ctx.ezpc_threshold.append(current_node.value)
            ctx.ezpc_depth.append(current_node.depth)
            queue.append(current_node.left)
            queue.append(current_node.right)
        else:
            ctx.ezpc_features.append(-1)
            ctx.ezpc_threshold.append(current_node.value)
            ctx.ezpc_depth.append(current_node.depth)


def parse_graphviz_to_ezpc_input(tree_file_path, task, scaling_factor):
    with open(tree_file_path, "r") as f:
        lines = f.readlines()
    lines = lines[1:]

    depth = 0
    nodes_this_tree = 0

    features = []
    threshold = []

    for i in range(len(lines)):
        curline = lines[i]
        # print("processing :", curline)
        start_location = curline.find('"')
        start_location += 1
        if start_location == 0:
            break
        nodes_this_tree += 1
        if curline[start_location] == "X":
            # This is an internal node
            end_location_feature = curline.find("]")
            start_location_th = curline.find("<=")
            end_location_th = curline.find("\\n")
            feature_val = int(curline[start_location + 2 : end_location_feature])
            threshold_val = float(curline[start_location_th + 3 : end_location_th])
            features.append(feature_val)
            threshold.append(threshold_val)
            # print("Internal Node")
            # print(feature_val)
            # print(threshold_val)
        else:
            # This is a leaf
            start_location_val = -1
            if task == "reg":
                start_location_val = curline.find("value =")
            else:
                start_location_val = curline.find("class =")
            assert start_location_val != -1, (
                "Task specified: " + task + " may be incorrect!"
            )
            end_location_val = curline.find('" filled')
            output_val = float(curline[start_location_val + 7 : end_location_val])
            features.append(-1)
            threshold.append(output_val)
            # print("Leaf Node")
            # print(output_val)

    class Context(object):
        def __init__(self):
            self.ctr = 0
            self.ezpc_features = []
            self.ezpc_threshold = []
            self.ezpc_depth = []
            self.max_depth = -1
            self.nodes_in_complete_tree = -1

    ctx = Context()
    root = fill_recur(ctx, features, threshold, 1)
    ctx.nodes_in_complete_tree = pow(2, ctx.max_depth) - 1
    # if nodes_in_complete_tree != nodes_this_tree:
    # print("[PADDING] Input tree not complete. Padding to make complete.")
    # else:
    # print("Input tree already complete. No need to pad.")

    pad_to_complete_tree(ctx, root)
    dump_complete_tree(ctx, root)

    model_weights = "weight_sf_" + str(scaling_factor) + ".inp"
    ezpc_tree_path = os.path.join(os.path.dirname(tree_file_path), model_weights)

    # print("Writing to " + ezpc_tree_path)
    # print("[FLOAT TO FIXED] Scaling by 2^" + str(scaling_factor) + " times")
    with open(ezpc_tree_path, "a") as output_file:
        for i in range(len(ctx.ezpc_features)):
            output_file.write(str(ctx.ezpc_features[i]) + "\n")
        for i in range(len(ctx.ezpc_threshold)):
            output_file.write(
                str(int(math.floor((2 ** scaling_factor) * ctx.ezpc_threshold[i])))
                + "\n"
            )

    return ctx.max_depth
