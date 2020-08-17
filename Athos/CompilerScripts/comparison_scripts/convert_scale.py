import sys
if __name__ == '__main__':
	assert(len(sys.argv) == 4)
	file_name = sys.argv[1]
	input_scale = int(sys.argv[2])
	output_scale = int(sys.argv[3])
	output_file_name = file_name + '_' + str(output_scale)
	output_file = open(output_file_name, "w")
	with open(file_name, "r") as a_file:
		for line in a_file:
			output = int((float(int(line.strip()))/(2**input_scale)) * (2**output_scale))
			output_file.write(str(output) + '\n')
	output_file.close()
