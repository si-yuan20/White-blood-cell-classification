# -*- coding: utf-8 -*-
"""
DEPRECATED — This file is a legacy benchmark script.
Please use the updated benchmark tool instead:

    python tools/benchmark_model.py --model_name all

This file is kept for backward compatibility only and will be removed
in a future version.
"""
import sys
import os

print("[DEPRECATED] test.py is deprecated.")
print("Please use: python tools/benchmark_model.py --model_name all")
print()

# Redirect to the real tool
benchmark_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "tools", "benchmark_model.py")
if os.path.isfile(benchmark_path):
    print(f"Running {benchmark_path} instead...")
    print()
    # Execute the benchmark script with forwarded args
    sys.argv[0] = benchmark_path
    # Add --model_name all if not specified
    if "--model_name" not in sys.argv:
        sys.argv = [benchmark_path] + sys.argv[1:] + ["--model_name", "all"]
    with open(benchmark_path) as f:
        exec(compile(f.read(), benchmark_path, "exec"))
else:
    print("ERROR: tools/benchmark_model.py not found.")
    sys.exit(1)
