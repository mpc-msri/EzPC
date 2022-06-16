#! /usr/bin/env python


# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import os
import argparse


def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-sdir",
        "--seedot_dir",
        type=str,
        metavar="",
        help="Location to SeeDot home dir",
    )
    parser.add_argument(
        "-rdir",
        "--results_dir",
        type=str,
        metavar="",
        help="Location to store the sirnn inference files.",
    )
    parser.add_argument(
        "-tdir",
        "--template_dir",
        type=str,
        metavar="",
        default="templates/",
        help="Location where the template file for running SIRNN inference are stored. \
                                (Required only if running this file from different directory).",
    )
    parser.add_argument(
        "-pdir",
        "--predict_dir",
        type=str,
        metavar="",
        help="Location to store predict.ezpc file.",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        choices=["Google-30"],
        default="Google-30",
        metavar="",
        help="The dataset to perform inference.",
    )

    parser.add_argument(
        "-sci",
        "--sci_build_location",
        type=str,
        default="",
        metavar="",
        help="The location where SCI is installed",
    )

    args = parser.parse_args()
    return args


def copyFiles(args):

    os.popen(
        "cp %s/EzPC/predict.ezpc %s/"
        % (os.path.abspath(args.seedot_dir), os.path.abspath(args.predict_dir))
    )
    os.popen(
        "cp -r %s/temp/Predictor/input/ %s/%s/"
        % (
            os.path.abspath(args.seedot_dir),
            os.path.abspath(args.results_dir),
            args.dataset,
        )
    )

    os.popen(
        "cp %s/temp/Predictor/model_fixed.h %s/%s/"
        % (
            os.path.abspath(args.seedot_dir),
            os.path.abspath(args.results_dir),
            args.dataset,
        )
    )
    os.popen(
        "cp %s/temp/Predictor/vars_fixed.h %s/%s/"
        % (
            os.path.abspath(args.seedot_dir),
            os.path.abspath(args.results_dir),
            args.dataset,
        )
    )
    os.popen(
        "cp %s/temp/Predictor/datatypes.h %s/%s/"
        % (
            os.path.abspath(args.seedot_dir),
            os.path.abspath(args.results_dir),
            args.dataset,
        )
    )

    os.popen(
        "cp %s/main.cpp %s/%s/"
        % (
            os.path.abspath(args.template_dir),
            os.path.abspath(args.results_dir),
            args.dataset,
        )
    )
    os.popen(
        "cp %s/predictors.h %s/%s/"
        % (
            os.path.abspath(args.template_dir),
            os.path.abspath(args.results_dir),
            args.dataset,
        )
    )

    os.popen(
        "cp %s/CMakeLists.txt_Dataset %s/%s/CMakeLists.txt"
        % (
            os.path.abspath(args.template_dir),
            os.path.abspath(args.results_dir),
            args.dataset,
        )
    )

    if not os.path.exists(
        os.path.join(os.path.abspath(args.results_dir), "CMakeLists.txt")
    ):
        os.popen(
            "cp %s/CMakeLists.txt %s/"
            % (os.path.abspath(args.template_dir), os.path.abspath(args.results_dir))
        )


def makeDatasetDir(args):
    os.makedirs(os.path.join(args.results_dir, args.dataset), exist_ok=True)


def resetMYINT(args):
    cur_dir = os.path.abspath(os.getcwd())
    os.chdir(os.path.join(args.results_dir, args.dataset))

    file = open("datatypes.h").read().split("\n")
    file[3] = file[3].replace("int16_t", "int32_t")

    f = open("datatypes.h", "w")
    for line in file:
        f.write(line + "\n")
    f.close()

    os.chdir(cur_dir)


def fixModelFixed(args):
    cur_dir = os.path.abspath(os.getcwd())
    os.chdir(os.path.join(args.results_dir, args.dataset))

    file = open("model_fixed.h").read()

    file = file.replace("][", "*")
    file = file.replace("seedot_fixed", "sirnn_fixed")
    f = open("model_fixed.h", "w")
    f.write(file)
    f.close()
    os.chdir(cur_dir)


def fixCMakeLists(args):
    cur_dir = os.path.abspath(os.getcwd())

    if args.sci_build_location == "":
        sci_build_location = os.path.join(os.path.abspath(args.results_dir), "build/")
    else:
        sci_build_location = os.path.abspath(args.sci_build_location)

    os.chdir(os.path.join(args.results_dir, args.dataset))

    file = open("CMakeLists.txt").read()
    file = file.replace("EXECUTALBLE_NAME", args.dataset)

    f = open("CMakeLists.txt", "w")
    f.write(file)
    f.close

    os.chdir("..")
    file = open("CMakeLists.txt").read()

    sub_dir_str = "add_subdirectory(%s)" % (args.dataset)

    file = file.replace("SCI_BUILD_LOCATION", '"%s"' % (sci_build_location))

    if file.find(sub_dir_str) == -1:
        file = file + "\n\n%s\n" % (sub_dir_str)
        f = open("CMakeLists.txt", "w")
        f.write(file)
        f.close()

    os.chdir(cur_dir)


def run(args):
    makeDatasetDir(args)
    copyFiles(args)
    resetMYINT(args)
    fixModelFixed(args)
    fixCMakeLists(args)


if __name__ == "__main__":
    args = parseArgs()
    run(args)
