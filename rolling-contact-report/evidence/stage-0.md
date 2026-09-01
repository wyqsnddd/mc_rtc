# Stage 0 baseline evidence

Date: 2026-09-01 (Asia/Shanghai)

This file records observed baseline results. A failure listed here is not presented as a rolling-contact regression.

## Revisions and environment

| Component | Observed revision/version |
| --- | --- |
| mc_rtc | `fc7df14a96f1273fb1aaff8e1a84ef514455d29c` (`master`; untracked report/user files present) |
| mc_rtc project version | 2.15.2 |
| Tasks source | `4d9fe01018ff0906475c829568b5d23b6705327a`; installed/build version 1.8.4 |
| TVM source | `e82b990e0b556efe33c1dc5fe2cae42bf423a19b`; installed version 0.9.5 |
| RBDyn | 1.9.5 |
| SpaceVecAlg | 1.2.10 |
| Eigen | 3.4.0 |
| CPU QP solver | eigen-qld 1.2.7 through the current Tasks/TVM CPU stack |
| ROS 2 | Jazzy (`ROS_DISTRO=jazzy`) |
| MuJoCo | archive `mujoco-3.4.0-linux-x86_64.tar.gz` found only in the read-only reference controller tree |
| mc_mujoco | not found in `PATH` or below `/home/yuquan/local` at audit time |
| CPU | AMD Ryzen 9 9950X, 16 cores/32 threads, one NUMA node, 64 MiB L3 |
| compiler | GCC 13.3.0 (Ubuntu 24.04) |
| CMake/CTest | 3.28.3 |
| build type | RelWithDebInfo |

The existing `build` links `libTasks.so.1` from `build/deps/tasks-system-install`, RBDyn/TVM/eigen-qld from
`/usr/local`, and system Boost 1.83. Its `libmc_solver.so` has no CUDA or Torch dependency.

## Commands and results

1. Existing build: `cmake --build build -j2` — exit 0, all configured targets built.
2. Fresh configure: `cmake -S . -B build-rolling -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
   -DENABLE_FAST_TESTS=ON` — exit 1 before compilation. `/usr/local` ndcurves unconditionally locates the Conda
   eigenpy package, whose Python 3.13 configuration requests Boost `python313`; CMake selected system Boost 1.83,
   which lacks that component. This is a mixed-prefix baseline issue, not a rolling source failure.
3. Sandboxed baseline: `ctest --test-dir build --output-on-failure -j2` — 32/89 passed; 57 failed/not run because tests
   attempted to write `/home/yuquan/.local/share/mc_rtc`, which is read-only in the workspace sandbox.
4. `XDG_DATA_HOME=/tmp/...` rerun — same 32/89 result; the robot cache path is explicitly based on the user data path
   already embedded by the build and did not honor this runtime override.
5. Unsandboxed rerun with `CUDA_VISIBLE_DEVICES=-1` — the cache became writable, but the existing cache contained
   incomplete JVRC convex-hull data. qhull reported missing hulls/internal output errors and 58 tests failed or were
   not run. This is retained as a pre-change baseline failure; no rolling code existed during any run.
6. `command -v mc_mujoco` — no result.

Focused framework/math tests that do not instantiate the affected cached JVRC model pass in the existing build,
including `testConfiguration`, `testConstraintSetLoader`, `testSolverBackend`, `testSolverTaskStorage`,
`test_mc_rbdyn_surface`, and the basic utility suites.

## CPU-only observation

All baseline commands used `CUDA_VISIBLE_DEVICES=-1` where runtime behavior was relevant. The active mc_rtc tree has
no confirmed rolling/GPU build option to disable, so no option name is invented. Production rolling targets will get
an automated `ldd`/dependency check rejecting CUDA, cuBLAS, cuSolver, Torch, or other GPU runtimes.

## Open Stage 0 gates

- Add the executable Tasks/TVM decision-vector inventory using the actual toy topology.
- Install or locate `mc_mujoco` and record its revision.
- Repair or isolate the pre-existing JVRC/qhull test cache for a green full-suite reference.
- Implement and prove the dynamic point/cone coefficient-update path selected in ADR-001.

Stage 0 remains **In progress** until these executable probes are complete. The source/API audit and architecture
decision are complete enough to begin the solver-independent geometry implementation without changing topology or
decision-vector semantics.
