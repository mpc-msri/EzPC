nt = 8

folder = "results_final_4june"
corr = 0
tot = 0
lab = "pos"
for i in range(nt):
    lines = open(folder + "/" + lab + "/thread-" + str(i)).readlines()
    for l in lines:
        # print(l.split(" ")[3])
        if lab == l.split(" ")[3].strip():
            corr += 1
        tot += 1

lab = "neg"
for i in range(nt):
    lines = open(folder + "/" + lab + "/thread-" + str(i)).readlines()
    for l in lines:
        # print(l.split(" ")[3])
        if lab == l.split(" ")[3].strip():
            corr += 1
        tot += 1

print(corr, tot)
print(corr / tot)
