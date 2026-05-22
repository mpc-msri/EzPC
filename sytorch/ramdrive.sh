#!/bin/bash

sudo mkdir /tmp/ramdisk
sudo chmod 777 /tmp/ramdisk
sudo mount -t tmpfs -o size=$1 myramdisk /tmp/ramdisk
mount | tail -n 1