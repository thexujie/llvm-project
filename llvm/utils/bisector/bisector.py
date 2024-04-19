#!/usr/bin/env python3

import subprocess
import sys
import os

LINKER=os.environ["BISECTOR_LINKER"]

# The bisector finds guilty translation units so we ignore link steps
if not "-c" in sys.argv:
  res = subprocess.run(LINKER.split() + sys.argv[1:])
  exit(res.returncode)

SAFE_COMPILER=os.environ["BISECTOR_SAFE_COMPILER"]
UNSAFE_COMPILER=os.environ["BISECTOR_UNSAFE_COMPILER"]

# List of bisector commands that will be run
CMD_LIST=os.environ["BISECTOR_CMD_LIST"]
if not os.path.exists(CMD_LIST):
  os.mknod(CMD_LIST)

# List of chunks that should use the unsafe tool
CHUNKS = os.environ["BISECTOR_CHUNKS"]

verbose=0
if os.environ["BISECTOR_VERBOSE"]:
  verbose=int(os.environ["BISECTOR_VERBOSE"])

def log(level=1, *args, **kwargs):
  if verbose >= level:
    print(*args, **kwargs)

# The signature is the working directory + the arguments passed to the bisector
cmd_signature = f"cd {os.getcwd()} && \"" + "\" \"".join(sys.argv) + "\""

if "BISECTOR_DUMP_CMD" in os.environ:
  with open(os.environ["BISECTOR_DUMP_CMD"], 'a') as f:
    f.write(cmd_signature)

# Start of the Chunks list parser
def consume_int():
  global CHUNKS
  idx = 0
  int_str = ''
  while len(CHUNKS) != 0 and ord(CHUNKS[0]) >= ord('0') and ord(CHUNKS[0]) <= ord('9'):
    idx += 1
    int_str += CHUNKS[0]
    CHUNKS = CHUNKS[1:]
  return int(int_str)

def consume_char(C):
  global CHUNKS
  if len(CHUNKS) != 0 and CHUNKS[0] == C:
    CHUNKS = CHUNKS[1:]
    return True
  return False

INT_SET = set()

while (1):
  Start = consume_int()
  if (consume_char('-')):
    End = consume_int()
    INT_SET |= set([I for I in range(Start, End + 1)])
  else:
    INT_SET |= {Start}
  
  if consume_char(':'):
    continue

  if len(CHUNKS) == 0:
    break
# End of the Chunks list parser
# The result of the chunk list is in INT_SET

args = sys.argv[1:]
found_signature = False
should_use_unsafe = False

# Traverse the CMD_LIST to look for the signature
idx = 0
with open(CMD_LIST) as file:
  for line in file:
    line = line[:-1]
    if cmd_signature == line:
      found_signature = True
      if idx in INT_SET:
        should_use_unsafe = True

      # Once we found the command we have nothing else to do
      break
    idx += 1

# If we didn't find the signature in the CMD_LIST file we add it to the CMD_LIST
if not found_signature:
  if idx in INT_SET:
    should_use_unsafe = True
  log(1, f"failed to find \"{cmd_signature}\" inside {CMD_LIST}")
  with open(CMD_LIST, "a") as file:
    file.write(cmd_signature)
    file.write("\n")

if should_use_unsafe:
  log(1, f"using unsafe for: {cmd_signature}")
  res = subprocess.run(UNSAFE_COMPILER.split() + args)
  exit(res.returncode)
else:
  log(1, f"using safe: {cmd_signature}")
  res = subprocess.run(SAFE_COMPILER.split() + args)
  exit(res.returncode)
