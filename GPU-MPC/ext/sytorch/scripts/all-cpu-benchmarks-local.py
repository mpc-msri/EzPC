import subprocess
import csv

mute = False

benchmarks = [
    'bert-tiny',
    'bert-base',
    'bert-large', # very large key
    'gpt2',
    'gptneo', # very large key
    "llama-7b", # very large key
    "llama-13b", # very large key
]

logfile1 = open("log1.log", 'a')
logfile2 = open("log2.log", 'a')
outcsv = open("results.csv", 'a')
outcsv.write("model,act_time,act_comm,softmax_time,softmax_comm,norm_time,norm_comm,total_time,total_comm\n")
outcsv.flush()

def run_seq(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=logfile1, stderr=logfile1)
    p.wait()


def run_par(cmd1, cmd2):
    p1 = subprocess.Popen(cmd1, shell=True, stdout=logfile1, stderr=logfile1)
    p2 = subprocess.Popen(cmd2, shell=True, stdout=logfile2, stderr=logfile2)
    p1.wait()
    p2.wait()

for b in benchmarks:
    print("[+] benchmarking " + b)
    print("[+] --- compiling...")
    run_seq('make benchmark-' + b)
    print("[+] --- running dealer...")
    run_seq(f'OMP_NUM_THREADS=4 ./benchmark-{b} 1')
    print("[+] --- running online phase...")
    run_par(f'OMP_NUM_THREADS=4 ./benchmark-{b} 2', f'OMP_NUM_THREADS=4 ./benchmark-{b} 3')

    total_time = 0.0
    total_comm = 0.0
    act_time = 0.0
    act_comm = 0.0
    softmax_time = 0.0
    softmax_comm = 0.0
    norm_time = 0.0
    norm_comm = 0.0
    with open('llama3.csv') as f:
        csvFile = csv.reader(f)
        header_skipped = False
        for lines in csvFile:
            if not header_skipped:
                header_skipped = True
                continue
            if lines[0].startswith('GeLU::'):
                act_time += float(lines[1])
                act_comm += float(lines[2])
            elif lines[0].startswith('LayerNorm::'):
                norm_time += float(lines[1])
                norm_comm += float(lines[2])
            elif lines[0].startswith('nExp::'):
                softmax_time += float(lines[1])
                softmax_comm += float(lines[2])
            elif lines[0].startswith('Softmax::'):
                softmax_time += float(lines[1])
                softmax_comm += float(lines[2])
            total_time += float(lines[1])
            total_comm += float(lines[2])
    run_seq(f"cp llama3.csv {b}.csv")
    print("[+] --- act time = " + str(act_time/1000.0) + " s")
    print("[+] --- act comm = " + str(act_comm/1024.0) + " GB")
    print("[+] --- softmax time = " + str(softmax_time/1000.0) + " s")
    print("[+] --- softmax comm = " + str(softmax_comm/1024.0) + " GB")
    print("[+] --- norm time = " + str(norm_time/1000.0) + " s")
    print("[+] --- norm comm = " + str(norm_comm/1024.0) + " GB")
    print("[+] --- online time = " + str(total_time/1000.0) + " s")
    print("[+] --- online comm = " + str(total_comm/1024.0) + " GB")
    outcsv.write(f"{b},{act_time/1000.0},{act_comm/1024.0},{softmax_time/1000.0},{softmax_comm/1024.0},{norm_time/1000.0},{norm_comm/1024.0},{total_time/1000.0},{total_comm/1024.0}\n")
    outcsv.flush()

logfile1.close()
logfile2.close()
outcsv.close()
