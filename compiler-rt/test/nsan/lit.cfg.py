# -*- Python -*-

import os

# Setup config name.
config.name = 'NSan' + config.name_suffix

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Test suffixes.
config.suffixes = ['.c', '.cc', '.test']

# C & CXX flags.
c_flags = ([config.target_cflags])

# Android doesn't want -lrt.
if not config.android:
  c_flags += ["-lrt"]

cxx_flags = (c_flags + config.cxx_mode_flags + ["-std=c++17"])

nsan_flags = ["-fsanitize=numerical", "-g",
              "-mno-omit-leaf-frame-pointer",
              "-fno-omit-frame-pointer"]

def build_invocation(compile_flags):
  return " " + " ".join([config.clang] + compile_flags) + " "

# Add substitutions.
config.substitutions.append(("%clang ", build_invocation(c_flags)))
config.substitutions.append(("%clang_nsan ", build_invocation(c_flags + nsan_flags)))
config.substitutions.append(("%clangxx_nsan ", build_invocation(cxx_flags + nsan_flags)))

# Platform-specific default NSAN for lit tests.
default_nsan_options = ''

config.environment['NSAN_OPTIONS'] = default_nsan_options
default_nsan_options += ':'
config.substitutions.append(('%env_nsan_options=',
                             'env NSAN_OPTIONS=' + default_nsan_options))

# NSan tests are currently supported on Linux only.
if config.host_os not in ['Linux']:
   config.unsupported = True
