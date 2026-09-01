#!/usr/bin/env python3

"""Reject unresolved or GPU runtime dependencies in rolling-contact ELF targets."""

import argparse
import json
import os
import pathlib
import re
import subprocess


GPU_RUNTIME = re.compile(
    r"(?:^|/)(?:lib)?(?:cuda|cudart|cublas|cusolver|cufft|curand|cusparse|"
    r"nvidia-ml|torch(?:_cuda)?|amdhip|hiprtc|opencl)[^/]*\.so",
    re.IGNORECASE,
)


def inspect(path, env):
    completed = subprocess.run(
        ["ldd", str(path)],
        check=False,
        capture_output=True,
        encoding="utf-8",
        env=env,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"ldd failed for {path}:\n{output}")
    unresolved = sorted(
        line.strip() for line in output.splitlines() if "not found" in line
    )
    gpu_dependencies = sorted(
        line.strip() for line in output.splitlines() if GPU_RUNTIME.search(line)
    )
    return {
        "path": str(path),
        "unresolved": unresolved,
        "gpu_runtime_dependencies": gpu_dependencies,
        "ldd": output.splitlines(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="+", type=pathlib.Path)
    parser.add_argument("--library-dir", action="append", default=[], type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    targets = [target.resolve(strict=True) for target in args.targets]
    library_directories = [path.resolve(strict=True) for path in args.library_dir]
    env = os.environ.copy()
    inherited = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(path) for path in library_directories]
        + ([inherited] if inherited else [])
    )
    results = [inspect(target, env) for target in targets]
    failures = [
        result
        for result in results
        if result["unresolved"] or result["gpu_runtime_dependencies"]
    ]
    report = {
        "schema": 1,
        "status": "fail" if failures else "pass",
        "device_contract": "CPU-only",
        "library_directories": [str(path) for path in library_directories],
        "targets": results,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    summary = {
        "status": report["status"],
        "device_contract": report["device_contract"],
        "target_count": len(results),
        "report": str(args.output.resolve()) if args.output else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(
            "rolling-contact target has unresolved or GPU runtime dependencies"
        )


if __name__ == "__main__":
    main()
