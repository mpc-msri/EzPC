fl = open('temp.txt', 'r')
temp = fl.readlines()
fl.close()

a = temp[0].split() 
container_id = a[8]

out = open('container_id.txt', 'w')
out.write(container_id)
out.close()
