# 
# Copyright:
# 
# Copyright (c) 2024 Microsoft Research
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import json
import os
import csv

# -- matplotlib stuff --

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../..')
from experiments.utils import run_one

def get_time(line):
    return round(float(line.split('=')[-1].split(' ')[0]) / 10**6, 3)

def get_comm(line):
    return round(float(line.split('(')[-1].split(' ')[0]), 3)

def run_perf(party, gpu, peer_ip, cpu_threads):
    for model in ['bert-tiny', 'bert-base', 'bert-large', 'gpt2', 'gpt-neo', 'gpt-neo-large', 'llama7b', 'llama13b']:
        cmd = "CUDA_VISIBLE_DEVICES={} ./sigma {} 128 {} {} {}".format(gpu, model, party, peer_ip, cpu_threads)
        log_dir = "output/P{}/models/{}-128/".format(party, model)
        run_one(cmd, log_dir, "logs.txt")

    stats = dict({'dealer': dict(), 'evaluator': dict()})
    for model in ['bert-tiny', 'bert-base', 'bert-large', 'gpt2', 'gpt-neo', 'gpt-neo-large', 'llama7b', 'llama13b']:
        stats['dealer'][model] = dict()
        stats['evaluator'][model] = dict()
        
        dealer_lines = open('output/P{}/models/{}-128/dealer.txt'.format(party, model)).readlines()
        stats['dealer'][model]['time'] = get_time(dealer_lines[0])
        stats['dealer'][model]['key_size'] = get_comm(dealer_lines[1])

        eval_lines = open('output/P{}/models/{}-128/evaluator.txt'.format(party, model)).readlines()
        stats['evaluator'][model]['gelu'] = dict()
        stats['evaluator'][model]['gelu']['time'] = get_time(eval_lines[6])
        stats['evaluator'][model]['gelu']['comm'] = get_comm(eval_lines[11])
        stats['evaluator'][model]['softmax'] = dict()
        stats['evaluator'][model]['softmax']['time'] = get_time(eval_lines[7])
        stats['evaluator'][model]['softmax']['comm'] = get_comm(eval_lines[12])
        stats['evaluator'][model]['layernorm'] = dict()
        stats['evaluator'][model]['layernorm']['time'] = get_time(eval_lines[8])
        stats['evaluator'][model]['layernorm']['comm'] = get_comm(eval_lines[13])
        stats['evaluator'][model]['total'] = dict()
        stats['evaluator'][model]['total']['time'] = get_time(eval_lines[0])
        stats['evaluator'][model]['total']['comm'] = get_comm(eval_lines[10])
    
    with open('output/P{}/Table3.json'.format(party), 'w') as outfile:
        table3 = dict()
        for tup in [('BERT-tiny', 'bert-tiny'), ('BERT-base', 'bert-base'), ('BERT-large', 'bert-large'), ('GPT2', 'gpt2'), ('GPT-Neo', 'gpt-neo'), ('Llama2-7B', 'llama7b'), ('Llama2-13B', 'llama13b')]:
            pretty_name, model = tup
            table3[pretty_name] = {
                'Activation': 
                {
                    'Time (s)': stats['evaluator'][model]['gelu']['time'], 
                    'Comm (GB)': stats['evaluator'][model]['gelu']['comm']
                },
                'Softmax': 
                {
                    'Time (s)': stats['evaluator'][model]['softmax']['time'], 
                    'Comm (GB)': stats['evaluator'][model]['softmax']['comm']
                },
                'Norm': 
                {   'Time (s)': stats['evaluator'][model]['layernorm']['time'], 
                    'Comm (GB)': stats['evaluator'][model]['layernorm']['comm']
                }
            }
        json.dump(table3, outfile, indent=4)
    
    with open('output/P{}/Table5.json'.format(party), 'w') as outfile:
        table5 = dict()
        for tup in [('BERT-tiny', 'bert-tiny'), ('BERT-base', 'bert-base'), ('BERT-large', 'bert-large'), ('GPT2', 'gpt2'), ('GPT-Neo', 'gpt-neo'), ('Llama2-7B', 'llama7b'), ('Llama2-13B', 'llama13b')]:
            pretty_name, model = tup
            table5[pretty_name] = {
                'Time (s)': stats['evaluator'][model]['total']['time'],
                'Comm (GB)': stats['evaluator'][model]['total']['comm']
            }
        json.dump(table5, outfile, indent=4)

    
    with open('output/P{}/Table9.json'.format(party), 'w') as outfile:
        table9 = dict()
        for tup in [('BERT-tiny', 'bert-tiny'), ('BERT-base', 'bert-base'), ('BERT-large', 'bert-large'), ('GPT2', 'gpt2'), ('GPT-Neo', 'gpt-neo'), ('Llama2-7B', 'llama7b'), ('Llama2-13B', 'llama13b')]:
            pretty_name, model = tup
            table9[pretty_name] = {
                'Key size (GB)': stats['dealer'][model]['key_size'],
                'Generation time (s)': stats['dealer'][model]['time'],
                'Online time (s)': stats['evaluator'][model]['total']['time']
            }
        json.dump(table9, outfile, indent=4)

    with open('output/P{}/Fig11_data.csv'.format(party),'w') as out_file:
        online_time = list(map(lambda model: stats['evaluator'][model]['total']['time'], ['gpt-neo', 'gpt-neo-large', 'llama7b', 'llama13b']))
        X = ('1.3', '2.7', '7', '13')
        plt.plot(X, online_time, marker='s', label='SIGMA-GPU')
        plt.legend(loc='upper left')
        plt.xlabel('Number of parameters (in billions)')
        plt.ylabel('Time (s)')
        plt.savefig("output/P{}/Fig11.png".format(party), dpi=300, bbox_inches='tight')
        plt.clf()
        
        writer = csv.writer(out_file)
        writer.writerow(['Number of parameters (in billions)','Time (s)'])
        for i in range(len(X)):
            writer.writerow((X[i], online_time[i]))


def run_table8(party, gpu, peer_ip, cpu_threads):
    for n_seq in [64, 128, 256, 512, 1024]:
        cmd = "CUDA_VISIBLE_DEVICES={} ./sigma gpt2 {} {} {} {}".format(gpu, n_seq, party, peer_ip, cpu_threads)
        log_dir = 'output/P{}/models/gpt2-{}/'.format(party, n_seq)
        run_one(cmd, log_dir, "logs.txt")

    with open('output/P{}/Table8.json'.format(party), 'w') as outfile:
        table8 = dict()
        for n_seq in [64, 128, 256, 512, 1024]:
            eval_lines = open('output/P{}/models/gpt2-{}/evaluator.txt'.format(party, n_seq)).readlines()
            table8[n_seq] = {
                'Time (s)': get_time(eval_lines[0]),
                'Comm (GB)': get_comm(eval_lines[10])
            }
        json.dump(table8, outfile, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Run artifact evaluation!')
    parser.add_argument('--n_seq', default=False, type=bool, help='Run Table 8.')
    parser.add_argument('--perf', default=False, type=bool, help='Run all performance experiments.')
    parser.add_argument('--all', default=False, type=bool, help='Run all experiments.')
    parser.add_argument('--party', default=0, type=int, help='Party to run (0/1).')

    args = parser.parse_args();
    global_config = None
    with open('config.json', 'r') as f:
        global_config = json.load(f)
    config = None

    if args.party == None:
        raise Exception("Must specify party")
    if args.party == 0:
        config = global_config['P0']
    else:
        config = global_config['P1']
    if args.all:
        run_perf(args.party, config['gpu'], config['peer'], config['cpu_threads'])
        run_table8(args.party, config['gpu'], config['peer'], config['cpu_threads'])
    elif args.perf:
        run_perf(args.party, config['gpu'], config['peer'], config['cpu_threads'])
    elif args.n_seq:
        run_table8(args.party, config['gpu'], config['peer'], config['cpu_threads'])

if __name__ == '__main__':
    main();
