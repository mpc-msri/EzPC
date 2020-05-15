f = open("debug/val_results.txt", 'r')
lines = f.readlines()
f.close()

lab_acc = [0, 0, 0]
lab_cnt = [0, 0, 0]
total_acc = 0
total_cnt = 0

for i in range(int(len(lines)/2)):
    if(lines[2*i][4:-1] == "COVID19"):
        lab_cnt[0] += 1
        if(lines[2*i+1][:-1] == "Covid19"):
            lab_acc[0] += 1
    elif(lines[2*i][4:-1] == "Normal"):
        lab_cnt[1] += 1
        if(lines[2*i+1][:-1] == "Normal"):
            lab_acc[1] += 1
    elif(lines[2*i][4:-1] == "Pneumonia"):
        lab_cnt[2] += 1
        if(lines[2*i+1][:-1] == "Pneumonia"):
            lab_acc[2] += 1

print(lab_acc)
print(lab_cnt)
total_acc = sum(lab_acc)
total_cnt = sum(lab_cnt)
total_acc /= total_cnt
lab_acc = [lab_acc[i] / lab_cnt[i] for i in range(3)]
print(lab_acc)
print(total_acc)
