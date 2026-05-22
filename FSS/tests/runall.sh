#!/bin/bash
for testname in $(ls -d */); do 
    set -e
    cd $testname
    # . config.sh

    echo -n "[+] compiling test..."
    fssc --bitlen ${BITLENGTH:='64'} main.ezpc &> /dev/null
    echo "done"

    echo -n "[+] running dealer..."
    ./main.out r=1 file=1 &> /dev/null
    echo "done"

    echo -n "[+] running p1..."
    cat input1.txt | ./main.out r=2 file=1 > tmp_output1.txt 2> /dev/null &
    p1pid=$!
    echo "done (PID = $p1pid)"
    sleep ${SLEEPTIME:='1'}
    echo -n "[+] running p2..."
    cat input2.txt | ./main.out r=3 file=1 > tmp_output2.txt 2> /dev/null
    echo "done"
    echo -n "[+] waiting for p1..."
    wait $p1pid
    echo "done"

    set +e
    cmp -s tmp_output1.txt output1.txt
    if [ $? -ne 0 ]; then
        echo "[-] test failed - compare output1.txt and tmp_output1.txt"
        exit 1
    fi

    cmp -s tmp_output2.txt output2.txt
    if [ $? -ne 0 ]; then
        echo "[-] test failed - compare output2.txt and tmp_output2.txt"
        exit 1
    fi

    echo "[+] test $testname passed"
    cd ..
done
