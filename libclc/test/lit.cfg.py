import os

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "libclc"

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [
    ".cl",
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.join(os.path.dirname(__file__))

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.test_run_dir, "test")

llvm_config.use_default_substitutions()

target = lit_config.params.get("target", "")
builtins = lit_config.params.get("builtins", "")

clang_flags = [
    "-fno-builtin",
    "-target",
    target,
    "-Xclang",
    "-mlink-builtin-bitcode",
    "-Xclang",
    os.path.join(config.libclc_lib_dir, builtins),
    "-nogpulib",
]

cpu = lit_config.params.get("cpu", "")
if cpu:
    clang_flags.append(f"-mcpu={cpu}")

llvm_config.use_clang(additional_flags=clang_flags)

tools = [
    "llvm-dis",
    "not",
]
tool_dirs = [config.llvm_tools_dir]

llvm_config.add_tool_substitutions(tools, tool_dirs)
