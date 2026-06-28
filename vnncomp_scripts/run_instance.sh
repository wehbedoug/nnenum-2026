#!/bin/bash
# example run_benchmark.sh script for VNNCOMP for nnenum
# six arguments, first is "v1", second is a benchmark category itentifier string such as "acasxu", third is path to the .onnx file, fourth is path to .vnnlib file, fifth is a path to the results file, and sixth is a timeout in seconds.
# Stanley Bak, Feb 2021

TOOL_NAME=nnenum
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4
RESULTS_FILE=$5
TIMEOUT=$6

echo "Running $TOOL_NAME on benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE', vnnlib file '$VNNLIB_FILE', results file $RESULTS_FILE, and timeout $TIMEOUT"

# setup environment variable for tool (doing it earlier won't be persistent with docker)"
DIR=$(dirname $(dirname $(realpath $0)))
export PYTHONPATH="$PYTHONPATH:$DIR/src"

# export PATH=${PATH}:$HOME/miniconda/bin

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# run the tool to produce the results file
####conda####
# conda_path = ${HOME}/anaconda3/bin
# py_pip_path = ${HOME}/anaconda3/bin

# conda_path = ${HOME}/miniconda/bin
# py_pip_path = ${HOME}/miniconda/envs/nnenumenv/bin # path for python, pip, grbprobe
# ${HOME}/anaconda3/envs/nnenumenv/bin/python -m nnenum.nnenum -o "$ONNX_FILE" -v "$VNNLIB_FILE" -t "$TIMEOUT" -f "$RESULTS_FILE" -s "$CATEGORY"
#SDW 2026-06-28;# python -m nnenum.nnenum -o "$ONNX_FILE" -v "$VNNLIB_FILE" -t "$TIMEOUT" -f "$RESULTS_FILE" -s "$CATEGORY"
#SDW 2026-06-28; commented the line above in favor of the one below, which uses miniconda.
${HOME}/miniconda/envs/nnenumenv/bin/python -m nnenum.nnenum -o "$ONNX_FILE" -v "$VNNLIB_FILE" -t "$TIMEOUT" -f "$RESULTS_FILE" -s "$CATEGORY"


####pipenv####
# export PATH="/root/.local/share/virtualenvs/toolkit-Z_JSe2nD/bin:$PATH"
# cd $DIR
# pipenv run python -m nnenum.nnenum -o "$ONNX_FILE" -v "$VNNLIB_FILE" -t "$TIMEOUT" -f "$RESULTS_FILE" -s "$CATEGORY"


####pip####
# python3 -m nnenum.nnenum -o "$ONNX_FILE" -v "$VNNLIB_FILE" -t "$TIMEOUT" -f "$RESULTS_FILE" -s "$CATEGORY"
