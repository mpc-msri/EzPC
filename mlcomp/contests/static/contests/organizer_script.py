#! env python
from __future__ import annotations
import json
import os
import shutil
from time import sleep
from enum import Enum
from urllib import request, parse
import subprocess
import numpy as np
import argparse
import os
import errno
from pathlib import Path
import coloredlogs
import logging
import traceback

# Create a logger object.
logging.basicConfig(
    filename="organizer_script_logs.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger("")
coloredlogs.install(level="DEBUG", logger=logger)

MPC_TIMEOUT = 60 * 60  # 1 hour

initial_config = {}
args = {}


class Config:
    def __init__(self, contest_id, access_key, website_url, num_testcases, labels_file):
        self.contest_id = contest_id
        self.access_key = access_key
        self.website_url = website_url
        self.num_testcases = num_testcases
        self.labels_file = labels_file

    def read_configuration() -> Config:
        print("Reading configuration'")
        config = initial_config
        config = Config(
            config["contest_id"],
            config["access_key"],
            config["website_url"],
            config["num_testcases"],
            config["labels_file"],
        )
        return config

    def next_submission_url(self) -> str:
        return (
            f"{self.website_url}/organizer/start_eval_next_submission/{self.contest_id}"
        )

    def get_upload_result_url(self, submission) -> str:
        return (
            f"{self.website_url}/organizer/upload_result/{submission['submission_id']}"
        )

    def get_failure_url(self, submission_id: int) -> str:
        return f"{self.website_url}/organizer/upload_failure/{submission_id}"


def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2  # copy R bits to X
    os.chmod(path, mode)


def concat_files(files: list[str], output_file: str):
    with open(output_file, "wb") as wfd:
        for f in files:
            with open(f, "rb") as fd:
                shutil.copyfileobj(fd, wfd)


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def last_line_of_file(filepath: str) -> str:
    last_line = ""
    with open(filepath) as f:
        for line in f:
            pass
        last_line = line
    return last_line


def delete_first_line(src: str, dest: str):
    with open(src, "r") as fin:
        data = fin.read().splitlines(True)
    with open(dest, "w") as fout:
        fout.writelines(data[1:])


def check_returncode(process):
    if process.returncode != 0:
        raise Exception("Return code was not 0")


class Judge:
    def __init__(self, config: Config):
        self.config = config
        self.eval_count: int = 0
        self.submission = None

    def cleanup(self):
        silentremove("labels.txt")

    def start_loop(self):
        logger.info("Starting evaluation loop.")
        while True:
            logger.info(f"Currently evaluating {self.eval_count} submission.")
            try:
                self.get_next_submission()
            except:
                logger.warning(
                    "Could not get next submission. Sleeping for 20s and continuing"
                )
                sleep(20)
                continue
            try:
                os.system("rm -rf output*")
                self.compile_model()
                os.system("rm -f processed_outputs.txt && touch processed_outputs.txt")
                for idx in range(self.config.num_testcases):
                    os.system("rm -f secret_shares.txt")
                    self.run_testcase(idx)
                    os.system("cat secret_shares.txt >> processed_outputs.txt")
                    logger.info("Sleeping for 3 seconds")
                    sleep(3)
                self.compare_labels()
                self.upload_results()
                logger.info(
                    f"Successfully completed {self.eval_count} evaluation(s). Starting next evaluation."
                )
            except Exception as e:
                logger.error(f"Error! %s", e, exc_info=1)
                self.upload_error(repr(e) + traceback.format_exc())
            self.eval_count += 1
            logger.info("Sleeping for 10s, and restarting loop for next submission.")
            self.submission = None
            sleep(10)

    def compile_model(self):
        logger.info("Compiling model")
        p = subprocess.run(
            [
                "python",
                "/ezpc_dir/EzPC/Athos/CompileONNXGraph.py",
                "--config",
                "config.json",
                "--role",
                "client",
            ]
        )
        check_returncode(p)

    def compare_labels(self):
        # Generate properly formatted file for compare labels
        with open("compare_labels.txt", "w") as f:
            f.write(f"{args.num_testcases}\n{args.num_classes}\n")
            labels_str = open(args.labels_file).read()
            f.write(f"{labels_str}")
            secret_shares_str = open("processed_outputs.txt").read()
            f.write(f"{secret_shares_str}\n")

        bin_url = f"{args.website_url}/static/contests/objects/compare_labels_nm"
        logger.info(f"Downloading label comparison binary from website  {bin_url}")
        subprocess.run(["curl", "--output", "compare_labels", bin_url])
        make_executable("./compare_labels")
        logger.info("Starting compare_labels script")
        with open("compare_labels.txt") as inp:
            with open("accuracy.txt", "w") as output:
                # compare_labels 2(server)/1(client) PORT [IP_ADDRESS]
                p = subprocess.run(
                    [
                        "./compare_labels",
                        "1",
                        f"{self.submission['server_port']}",
                        f"{self.submission['server_ip']}",
                    ],
                    stdin=inp,
                    stdout=output,
                )
                check_returncode(p)

    def run_testcase(self, idx: int):
        logger.info(f"Running testcase {idx}")
        dataset_file = os.path.join(initial_config["dataset_path"], str(idx))
        p = subprocess.run(
            ["python", "pre_process.py", dataset_file, "input.npy"], timeout=MPC_TIMEOUT
        )
        check_returncode(p)
        p = subprocess.run(
            [
                "python",
                "/ezpc_dir/EzPC/Athos/CompilerScripts/convert_np_to_fixedpt.py",
                "--inp",
                "input.npy",
                "--output",
                "processed_input.fixedpt",
                "--config",
                "config.json",
            ]
        )
        check_returncode(p)

        with open("processed_input.fixedpt") as inp:
            with open(f"output_{idx}.txt", "w") as out:
                p = subprocess.run(
                    [
                        "./model_SCI_OT.out",
                        "r=2",
                        f"ip={self.submission['server_ip']}",
                        f"port={self.submission['server_port']}",
                    ],
                    stdin=inp,
                    stdout=out,
                    timeout=MPC_TIMEOUT,
                )
                check_returncode(p)

    def get_next_submission(self):
        logger.info("Getting next submission")
        self.submission = None
        url = self.config.next_submission_url()
        subprocess.run(["curl", url, "--output", "submission.json"])
        sub_json_file = open("submission.json")
        self.submission = json.load(sub_json_file)
        logger.info(f"Submission:  {self.submission}")
        subprocess.run(
            ["curl", self.submission["model_path"], "--output", "optimised_model.onnx"]
        )
        subprocess.run(
            ["curl", self.submission["config_path"], "--output", "config.json"]
        )
        subprocess.run(
            [
                "curl",
                self.submission["pre_process_script_path"],
                "--output",
                "pre_process.py",
            ]
        )
        logger.info("Getting next submission done")

    def compute_score(self):
        last_line = last_line_of_file("accuracy.txt")
        return int(last_line)

    # The result is available with client ( organizer )

    def upload_results(self):
        logger.info("Computing score")
        score = self.compute_score()
        logger.info(f"Calculated score as  {score}")
        logger.info("Uploading results to website")
        os.system("rm -f output.txt")
        os.system("cat output*.txt > output.txt")
        with open("output.txt") as out:
            data = out.read()
            score = self.compute_score()
            upload_data = {
                "contest_id": self.config.contest_id,
                "submission_id": self.submission["submission_id"],
                "output": "It is usually very long",
                "score": score,
                "eval_count": self.eval_count,
                "access_key": self.config.access_key,
            }
            url = self.config.get_upload_result_url(self.submission)
            data = parse.urlencode(upload_data).encode()
            req = request.Request(url, data=data)
            response = request.urlopen(req)
            logger.info(f"Server response, {response}")
        logger.info("Done uploading results")

    def upload_error(self, error: str):
        if self.submission is None:
            return

        try:
            upload_data = {
                "contest_id": args.contest_id,
                "submission_id": self.submission["submission_id"],
                "access_key": self.config.access_key,
                "error": error,
            }
            data = parse.urlencode(upload_data).encode()
            url = self.config.get_failure_url(self.submission["submission_id"])
            req = request.Request(url, data=data)
            request.urlopen(req)
        finally:
            logger.info("Uploaded failure to website")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for contest organizer.")
    parser.add_argument(
        "--contest_id", type=int, required=True, help="Integer ID of the contest"
    )
    parser.add_argument(
        "--access_key", type=str, required=True, help="Access key for the contest"
    )
    parser.add_argument(
        "--website_url",
        type=str,
        required=True,
        help="Complete URL/IP address with port of the  contest website",
    )
    parser.add_argument(
        "--num_testcases", type=int, required=True, help="Number of testcases"
    )
    parser.add_argument(
        "--num_classes", type=int, required=True, help="Number of label classes"
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        required=True,
        help="Path to text file containing scaled labels",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to dataset folder"
    )

    args = parser.parse_args()
    initial_config["contest_id"] = args.contest_id
    initial_config["access_key"] = args.access_key
    initial_config["website_url"] = args.website_url
    initial_config["num_testcases"] = args.num_testcases
    initial_config["num_classes"] = args.num_classes
    initial_config["labels_file"] = args.labels_file
    initial_config["dataset_path"] = args.dataset_path
    logger.info(f"Read configuration: {initial_config}")
    logger.info("Organizer ( client ) script starting")
    config = Config.read_configuration()
    judge = Judge(config)
    judge.start_loop()

# Sample arguments to run this script
# rm -f organizer_script.py && curl http://127.0.0.1:8000/static/contests/scripts/organizer_script.py --output organizer_script.py && python organizer_script.py --contest_id 4 --access_key defaultkey --website_url http://127.0.0.1:8000 --num_testcases 3 --labels_file labels.txt --dataset_path dataset --num_classes 5
