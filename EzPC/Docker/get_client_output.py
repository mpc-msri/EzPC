fl = open('../../../ABY-latest/ABY/build/bin/client_output.txt', 'r')

full_output = fl.readlines()
fl.close()

sf_file = open('../EzPC/seclud_random_forest/scaling_factor.txt', 'r')
sf = sf_file.readlines()
sf = int(sf[0])
sf_file.close()

trees_file = open('../EzPC/seclud_random_forest/decision_tree_stat.txt', 'r')
trees = trees_file.readlines()
trees = int(trees[0])
trees_file.close()

ans = int(full_output[-1])
ans = (ans*1.0)/(sf*trees*1.0)
print("[INFERENCE OUTPUT]: Client's final output of inference query is:")
print(ans)
print("[ENDING]: Inference task completed securely")
