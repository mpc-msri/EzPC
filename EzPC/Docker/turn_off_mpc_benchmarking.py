aby = open("../../../ABY-latest/ABY/src/abycore/ABY_utils/ABYconstants.h")

lines = aby.readlines()

aby.close()

print(lines[32])
print(lines[33])

lines[32] = lines[32][:31] + " 0" + lines[32][33:]
lines[33] = lines[33][:33] + " 0" + lines[33][35:]

print(lines[32])
print(lines[33])

new_aby = open("updated_ABYconstants.h", 'w')
for i in range(len(lines)):
    new_aby.write(lines[i])
new_aby.close()
