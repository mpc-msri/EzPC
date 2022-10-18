perf record -a -F 999 -g -- ./build/fptraining
perf script -i perf.data > out.perf
./.perf/foldperf.pl out.perf > out.folded
./.perf/flamegraph.pl out.folded > graph.svg
