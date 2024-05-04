#!/usr/bin/bash
# This is test / demo of the bisector + delta-driver
# run: ./delta-driver ./llvm/utils/bisector/bisector_demo.sh 0-5
# The execution of the delta-driver should finish by:
# Minimal Chunks = 2:4
# because of the artificial condition at the bottom of the scirpt

set -e

# Configure the bisector.py with environement variable

export BISECTOR_CHUNKS=$1
# File used to store all commands. line number is used as id in the chunk list by bisector.py
export BISECTOR_CMD_LIST=$0.cmd_list
# In real word example the safe and unsafe compiler should be under ccache
export BISECTOR_SAFE_COMPILER="/usr/bin/echo safe"
export BISECTOR_UNSAFE_COMPILER="/usr/bin/echo unsafe"
# There is no link step in out fake build
export BISECTOR_LINKER=
export BISECTOR_VERBOSE=2

TMP_FILE=$0.tmp

rm -rf $TMP_FILE

# This simulate a build system calling with 6 translation units
for i in $(seq 0 5); do
  # -c here is to prevent the call from being interpreted as a link step
  ./llvm/utils/bisector/bisector.py -c $i >> $TMP_FILE
done

# This simulate the test-suite of the software trying to reproduce the bug
# We fail if translation unit 2 or 4 used the safe compiler
grep -w "safe -c 2" $TMP_FILE || grep -w "safe -c 4" $TMP_FILE
