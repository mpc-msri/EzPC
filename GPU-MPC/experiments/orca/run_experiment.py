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

import subprocess
from pathlib import Path
import argparse
import json
import time
import os
import csv

# -- matplotlib stuff --

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../..')
from experiments.utils import run_seq, run_parallel, run_one, remove_key

def run_fig_helper(party, dealer_gpu, eval_gpu, dealer_key_dir, peer_ip, exp_name, fig_name, loss_dir, dataset):
    dealer_cmd = 'CUDA_VISIBLE_DEVICES={} ./orca_dealer {} {} {}'.format(dealer_gpu, party, exp_name, dealer_key_dir)
    eval_cmd = 'CUDA_VISIBLE_DEVICES={} ./orca_evaluator {} {} {} {}'.format(eval_gpu, party, peer_ip, exp_name, dealer_key_dir)

    log_dir = 'output/P{}/{}/logs/'.format(party, fig_name)
    run_parallel(dealer_cmd, eval_cmd, log_dir)
    key_file = '{}_training_key{}.dat'.format(exp_name.split('-')[0], party)
    print("Key file={}".format(key_file))
    remove_key(dealer_key_dir, key_file)

    loss = list(map(lambda x: float(x), open('output/P{}/training/loss/{}/loss.txt'.format(party, loss_dir)).readlines()))
    X = list(map(lambda x: 10 * (x + 1), range(len(loss))))
    plt.plot(X, loss)
    plt.title("{} on {}".format(exp_name.split('-')[0], dataset))
    plt.xlabel("Iterations")
    plt.ylabel("Cross-entropy loss")
    plt.savefig("output/P{}/{}/{}.png".format(party, fig_name, fig_name), dpi=300, bbox_inches='tight')
    plt.clf()
    with open('output/P{}/{}/loss.csv'.format(party, fig_name),'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(['Iteration','Cross-Entropy Loss'])
        for i in range(len(X)):
            writer.writerow((X[i], loss[i]))


def run_figure(fig_number, party, dealer_gpu, eval_gpu, dealer_key_dir, peer_ip):
    print("Generating Figure {}".format(fig_number))
    if fig_number == '5a':
        run_fig_helper(party, dealer_gpu, eval_gpu, dealer_key_dir, peer_ip, 'CNN2-loss', 'Fig5a', 'CNN2-1e-60b', 'MNIST')
    elif fig_number == '5b':
        run_fig_helper(party, dealer_gpu, eval_gpu, dealer_key_dir, peer_ip, 'CNN3-2e-loss', 'Fig5b', 'CNN3-2e-50b', 'CIFAR-10')
    else:
        print('unrecognized figure number', fig_number)
 

def run_table3(party, dealer_gpu, eval_gpu, dealer_key_dir, peer_ip):
    log_dir = 'output/P{}/Table3/logs/'.format(party)
    table = dict()
    for exp in ['CNN2', 'CNN3-2e', 'CNN3-5e']:
    # for exp in ['CNN2']:
        dealer_cmd = "CUDA_VISIBLE_DEVICES={} ./orca_dealer {} {} {}".format(dealer_gpu, party, exp, dealer_key_dir)
        eval_cmd = "CUDA_VISIBLE_DEVICES={} ./orca_evaluator {} {} {} {}".format(eval_gpu, party, peer_ip, exp, dealer_key_dir)
        log_dir = "output/P{}/Table3/logs/{}/".format(party, exp)
        if exp == 'CNN2':
            run_seq(dealer_cmd, eval_cmd, log_dir)
        else:
            run_parallel(dealer_cmd, eval_cmd, log_dir)
        key_file = '{}_training_key{}.dat'.format(exp.split('-')[0], party)
        remove_key(dealer_key_dir, key_file)

    for tup in [('CNN2-1e-1b', 1), ('CNN3-2e-20b', 40), ('CNN3-5e-20b', 100)]:
    # for tup in [('CNN3-5e-20b', 1)]:
        exp, blocks = tup
        network = exp.split('-')[0]
        training_stats = list(map(lambda x: x.split(":")[-1], open("output/P{}/training/{}.txt".format(party, exp)).readlines()))
        time = float(training_stats[0])
        comm = float(training_stats[1])
        key_read_time = float(training_stats[3])
        compute_time = float(training_stats[4])
        time -= (blocks - 1) * min(key_read_time, compute_time)
        network = '-'.join(exp.split('-')[:-1])
        table[network] = dict()
        table[network]['Epochs'] = int(exp.split('-')[1][0])
        table[network]['Time (min)'] = round(time / (1000 * 60), 2)
        table[network]['Comm (GB)'] = round(comm / (1024**3), 2)
        lines = open('output/P{}/training/loss/{}/accuracy.txt'.format(party, exp)).readlines()
        accuracy = 0.0
        if len(lines) > 0:
            accuracy = float(lines[-1])
        table[network]['Accuracy'] = round(accuracy, 2)
    
    with open('output/P{}/Table3/Table3.json'.format(party), 'w') as outfile:
        json.dump(table, outfile, indent=4)


def run_table4(party, dealer_gpu, eval_gpu, dealer_key_dir, peer_ip):
    log_dir = 'output/P{}/Table4/logs/'.format(party)

    for network in ['P-SecureML', 'P-LeNet', 'P-AlexNet', 'P-VGG16']:
        dealer_cmd = "CUDA_VISIBLE_DEVICES={} ./orca_dealer {} {} {}".format(dealer_gpu, party, network, dealer_key_dir)
        eval_cmd = "CUDA_VISIBLE_DEVICES={} ./orca_evaluator {} {} {} {}".format(eval_gpu, party, peer_ip, network, dealer_key_dir)
        run_seq(dealer_cmd, eval_cmd, log_dir)
        key_file = '{}_training_key{}.dat'.format(network, party)
        remove_key(dealer_key_dir, key_file)

    for network in ['P-SecureML', 'P-LeNet', 'P-AlexNet', 'P-VGG16']:
        dealer_cmd = "CUDA_VISIBLE_DEVICES={} ./piranha {} {} {} {}".format(dealer_gpu, network, 0, party, dealer_key_dir)
        eval_cmd = "CUDA_VISIBLE_DEVICES={} ./piranha {} {} {} {} {}".format(eval_gpu, network, 1, party, dealer_key_dir, peer_ip)
        run_seq(dealer_cmd, eval_cmd, log_dir)
        key_file = '{}_inference_key{}.dat'.format(network, party)
        remove_key(dealer_key_dir, key_file)
    
    table = dict()
    for network in ['P-SecureML', 'P-LeNet', 'P-AlexNet', 'P-VGG16']:
        table[network] = {'Training': dict(), 'Inference': dict()}

        training_stats = list(map(lambda x: x.split(":")[-1], open('output/P{}/training/{}.txt'.format(party, network)).readlines()))
        training_time = training_stats[-4]
        training_comm = training_stats[-1]
        
        table[network]['Training']['Time (ms)'] = round(float(training_time), 2)
        table[network]['Training']['Comm (MB)'] = round(float(training_comm) / 1024**2, 2)

        inference_stats = list(map(lambda x: x.split("=")[-1], open('output/P{}/inference/{}.txt'.format(party, network)).readlines()))
        inference_time = inference_stats[0]
        inference_comm = inference_stats[1]
        table[network]['Inference']['Time (ms)'] = round(float(inference_time) / 1000, 2)
        table[network]['Inference']['Comm (MB)'] = round(float(inference_comm) / 1024**2, 2)

    with open('output/P{}/Table4/Table4.json'.format(party), 'w') as outfile:
        json.dump(table, outfile, indent=4)


def run_table6(party, dealer_gpu, eval_gpu, dealer_key_dir, peer_ip):
    log_dir = 'output/P{}/Table6/logs/'.format(party)

    for tup in [('CNN2', '-perf'), ('ModelB', ''), ('AlexNet', ''), ('CNN3', '-perf')]:
        network, suffix = tup         
        dealer_cmd = "CUDA_VISIBLE_DEVICES={} ./orca_dealer {} {}{} {}".format(dealer_gpu, party, network, suffix, dealer_key_dir)
        eval_cmd = "CUDA_VISIBLE_DEVICES={} ./orca_evaluator {} {} {}{} {}".format(eval_gpu, party, peer_ip, network, suffix, dealer_key_dir)
        run_seq(dealer_cmd, eval_cmd, log_dir)
        key_file = '{}_training_key{}.dat'.format(network, party)
        remove_key(dealer_key_dir, key_file)
    
    table = dict()
    for network in ['CNN2', 'ModelB', 'AlexNet', 'CNN3']:
        stats = list(map(lambda x: x.split(":")[-1], open("output/P{}/training/{}.txt".format(party, network)).readlines()))
        time = stats[-4]
        comm = stats[-1]
        table[network] = dict()
        table[network]['Time (s)'] = round(float(time) / 1000, 2)
        table[network]['Comm (MB)'] = round(float(comm) / 1024**2, 2)
    
    with open('output/P{}/Table6/Table6.json'.format(party), 'w') as outfile:
        json.dump(table, outfile, indent=4)



def run_table7(party, dealer_gpu, eval_gpu, dealer_key_dir, peer_ip):
    log_dir = 'output/P{}/Table7/logs/'.format(party)

    for network in ['CNN2', 'CNN3']:
        dealer_cmd = "CUDA_VISIBLE_DEVICES={} ./orca_dealer {} {}-perf {}".format(dealer_gpu, party, network, dealer_key_dir)
        eval_cmd = "CUDA_VISIBLE_DEVICES={} ./orca_evaluator {} {} {}-perf {}".format(eval_gpu, party, peer_ip, network, dealer_key_dir)
        run_seq(dealer_cmd, eval_cmd, log_dir)
        key_file = '{}_training_key{}.dat'.format(network, party)
        remove_key(dealer_key_dir, key_file)

    for network in ['CNN2', 'CNN3']:
        bw = 64
        scale = 24
        dealer_cmd = "CUDA_VISIBLE_DEVICES={} ./orca_inference {} {} {} {} {} {}".format(dealer_gpu, network, bw, scale, 0, party, dealer_key_dir)
        eval_cmd = "CUDA_VISIBLE_DEVICES={} ./orca_inference {} {} {} {} {} {} {}".format(eval_gpu, network, bw, scale, 1, party, dealer_key_dir, peer_ip)
        run_seq(dealer_cmd, eval_cmd, log_dir)
        key_file = '{}_{}_{}_inference_key{}.dat'.format(network, bw, scale, party)
        remove_key(dealer_key_dir, key_file)

    table = dict()
    for network in ['CNN2', 'CNN3']:
        bw = 64
        scale = 24
        table[network] = {'Training': dict(), 'Inference': dict()}

        training_stats = list(map(lambda x: x.split(":")[-1], open('output/P{}/training/{}.txt'.format(party, network)).readlines()))
        training_time = training_stats[-4]        
        table[network]['Training']['Time (s)'] = round(float(training_time) / 1000, 2)
        
        inference_stats = list(map(lambda x: x.split("=")[-1], open('output/P{}/inference/{}_{}_{}.txt'.format(party, network, bw, scale)).readlines()))
        inference_time = inference_stats[0]        
        table[network]['Inference']['Time (s)'] = round(float(inference_time) / 1000**2, 2)

    with open('output/P{}/Table7/Table7.json'.format(party), 'w') as outfile:
        json.dump(table, outfile, indent=4)
    

def run_table8(party, dealer_gpu, eval_gpu, dealer_key_dir, peer_ip):
    log_dir = 'output/P{}/Table8/logs/'.format(party)
    table = dict()
    for network in ['CNN2', 'CNN3']:
        dealer_cmd = "CUDA_VISIBLE_DEVICES={} ./orca_dealer {} {}-perf {}".format(dealer_gpu, party, network, dealer_key_dir)
        run_one(dealer_cmd, log_dir)
        key_file = '{}_training_key{}.dat'.format(network, party)
        remove_key(dealer_key_dir, key_file)
        table[network] = dict()
        key_size = float(open('output/P{}/training/keysize/{}.txt'.format(party, network)).readlines()[0])
        table[network]['Training (GB)'] = round(float(key_size) / 1024**3, 2)

    for network in ['CNN2', 'CNN3']:
        bw = 64
        scale = 24
        dealer_cmd = "CUDA_VISIBLE_DEVICES={} ./orca_inference {} {} {} {} {} {}".format(dealer_gpu, network, bw, scale, 0, party, dealer_key_dir)
        run_one(dealer_cmd, log_dir)

        key_file = '{}_{}_{}_inference_key{}.dat'.format(network, bw, scale, party)
        key_size = os.path.getsize(dealer_key_dir + key_file) 
        table[network]['Inference (GB)'] = round(float(key_size) / 1024**3, 2)
        remove_key(dealer_key_dir, key_file)
    
    with open('output/P{}/Table8/Table8.json'.format(party), 'w') as outfile:
        json.dump(table, outfile, indent=4)
    


def run_table9(party, dealer_gpu, eval_gpu, dealer_key_dir, peer_ip):
    log_dir = 'output/P{}/Table9/logs/'.format(party)

    for tup in [('VGG16', 64, 24, ''), ('ResNet18', 64, 24, ''), ('ResNet50', 64, 24, ''), ('ResNet50', 37, 12, ''), ('VGG16', 32, 12, '_u32'), ('ResNet18', 32, 10, '_u32')]:
        network, bw, scale, bin_suffix = tup
        dealer_cmd = "CUDA_VISIBLE_DEVICES={} ./orca_inference{} {} {} {} {} {} {}".format(dealer_gpu, bin_suffix, network, bw, scale, 0, party, dealer_key_dir)
        eval_cmd = "CUDA_VISIBLE_DEVICES={} ./orca_inference{} {} {} {} {} {} {} {}".format(eval_gpu, bin_suffix, network, bw, scale, 1, party, dealer_key_dir, peer_ip)
        run_seq(dealer_cmd, eval_cmd, log_dir)
        key_file = '{}_{}_{}_inference_key{}.dat'.format(network, bw, scale, party)
        remove_key(dealer_key_dir, key_file)

    table = dict()
    for tup in [('VGG16', 64, 24), ('ResNet50', 64, 24), ('ResNet18', 64, 24), ('ResNet50', 37, 12), ('VGG16', 32, 12), ('ResNet18', 32, 10)]:
        network, bw, scale = tup
        if not network in table:
            table[network] = dict()
        table[network][bw] = dict()
        table[network][bw][scale] = dict()
        inference_stats = list(map(lambda x: x.split("=")[-1], open('output/P{}/inference/{}_{}_{}.txt'.format(party, network, bw, scale)).readlines()))
        inference_time = inference_stats[0]
        table[network][bw][scale]['Time (s)'] = round(float(inference_time) / 1000**2, 3)
   
    with open('output/P{}/Table9/Table9.json'.format(party), 'w') as outfile:
        json.dump(table, outfile, indent=4)


def run_table(table_number, party, dealer_gpu, eval_gpu, dealer_key_dir, peer_ip):
    print("Generating Table {}".format(table_number))
    if table_number == 3:
        run_table3(party, dealer_gpu, eval_gpu, dealer_key_dir, peer_ip)
    elif table_number == 4:
        run_table4(party, dealer_gpu, eval_gpu, dealer_key_dir, peer_ip)
    elif table_number == 6:
        run_table6(party, dealer_gpu, eval_gpu, dealer_key_dir, peer_ip)
    elif table_number == 7:
        run_table7(party, dealer_gpu, eval_gpu, dealer_key_dir, peer_ip)
    elif table_number == 8:
        run_table8(party, dealer_gpu, eval_gpu, dealer_key_dir, peer_ip)
    elif table_number == 9:
        run_table9(party, dealer_gpu, eval_gpu, dealer_key_dir, peer_ip)
    else:
        print('unrecognized table number', table_number)


def main():
    parser = argparse.ArgumentParser(description='Run artifact evaluation!')
    parser.add_argument('--figure', default=None, help='Figure # to run.')
    parser.add_argument('--table', default=None, type=int, help='Table # to run.')
    parser.add_argument('--all', default=None, help='Run all experiments.')
    parser.add_argument('--party', default=None, type=int, help='Party to run (0/1).')

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
    dealer_config = config['dealer']
    eval_config = config['evaluator']

    if args.all:
        for i in [4, 6, 7, 8, 9]:
            run_table(i, args.party, dealer_config['gpu'], eval_config['gpu'], dealer_config['key_dir'], eval_config['peer'])
        
        for i in ['5a', '5b']:
            run_figure(i, args.party, dealer_config['gpu'], eval_config['gpu'], dealer_config['key_dir'], eval_config['peer'])
        
        run_table(3, args.party, dealer_config['gpu'], eval_config['gpu'], dealer_config['key_dir'], eval_config['peer'])
    # Handle figure experiments
    elif args.figure:
        run_figure(args.figure, args.party, dealer_config['gpu'], eval_config['gpu'], dealer_config['key_dir'], eval_config['peer'])
        
    # Handle tables
    elif args.table:
        run_table(args.table, args.party,  dealer_config['gpu'], eval_config['gpu'], dealer_config['key_dir'], eval_config['peer'])

if __name__ == '__main__':
    main();
