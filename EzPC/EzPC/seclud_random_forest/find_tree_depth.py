import sys

features = sys.argv[1]

fl = open("decision_tree_stat.txt", 'r')
values = fl.readlines()
fl.close()

print(values)
ans = [values[0][:-1]]
ans.append(0)
for i in range(len(values)-1):
    ans[1] = max(ans[1], int(values[i+1][:-1]))
print(ans)

fl = open("decision_tree_stat1.txt", 'w')
fl.write(ans[0] + '\n')
fl.write(str(ans[1]) + '\n')
fl.write(features)
fl.close()

