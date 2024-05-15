import subprocess
from pathlib import Path
import json
import time
import os
import csv

def run_parallel(dealer_cmd, eval_cmd, log_dir):
    dealer = None
    evaluator = None
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    dealer_log = log_dir + "dealer.log"
    eval_log = log_dir + "eval.log"
    try:
        print('Running command={}'.format(dealer_cmd))
        with open(dealer_log, 'a') as dealer_file:
            dealer = subprocess.Popen(dealer_cmd, shell=True, stdout=dealer_file, stderr=dealer_file)

        print('Running command={}'.format(eval_cmd))
        with open(eval_log, 'a') as eval_file:
            evaluator = subprocess.Popen(eval_cmd, shell=True, stdout=eval_file, stderr=eval_file) 
    except:
        if dealer:
            dealer.terminate()
            dealer.wait()
        if evaluator:
            evaluator.terminate()
            evaluator.wait()
        raise Exception("Something went wrong. Please check the logs.")
    
    dealer_done = False
    eval_done = False
    while True:
        time.sleep(60)
        dealer_out = dealer.poll()
        # print("Dealer out={}".format(dealer_out))
        if dealer_out is not None:
            if dealer_out > 0:
                print("Killing evaluator.")
                evaluator.terminate()
                evaluator.wait()
                raise Exception("Dealer did not run properly. Check logs for errors.")
            else:
                dealer_done = True
        eval_out = evaluator.poll()
        # print("Eval out={}".format(dealer_out))
        if eval_out is not None:
            if eval_out > 0:
                print("Killing dealer.")
                dealer.terminate()
                dealer.wait()
                raise Exception("Evaluator did not run properly. Check logs for errors.")
            else:
                eval_done = True 
        if dealer_done and eval_done:
            break


def run_seq(dealer_cmd, eval_cmd, log_dir):
    dealer = None
    evaluator = None
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    dealer_log = log_dir + "dealer.log"
    print('Running command={}'.format(dealer_cmd))
    with open(dealer_log, 'a') as dealer_file:
        dealer = subprocess.run(dealer_cmd, shell=True, stdout=dealer_file, stderr=dealer_file, check=True)
        if dealer.returncode:
            raise Exception("Dealer did not run properly. Check logs for errors.")

    eval_log = log_dir + "eval.log"    
    print('Running command={}'.format(eval_cmd))
    with open(eval_log, 'a') as eval_file:
        evaluator = subprocess.run(eval_cmd, shell=True, stdout=eval_file, stderr=eval_file, check=True) 
        if evaluator.returncode:
            raise Exception("Evaluator did not run properly. Check logs for errors.")

def run_one(dealer_cmd, log_dir):
    dealer = None
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    dealer_log = log_dir + "dealer.log"
    print('Running command={}'.format(dealer_cmd))
    with open(dealer_log, 'a') as dealer_file:
        dealer = subprocess.run(dealer_cmd, shell=True, stdout=dealer_file, stderr=dealer_file, check=True)
        if dealer.returncode:
            raise Exception("Dealer did not run properly. Check logs for errors.")


def remove_key(key_dir, key_file):
    key_path = key_dir + key_file
    print("Removing key={}".format(key_path))
    if os.path.exists(key_path):
        os.remove(key_path)
