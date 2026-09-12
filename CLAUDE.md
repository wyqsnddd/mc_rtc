# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` holds the coding style, testing and commit conventions for this repo — read it too; this file
does not repeat them.

## Build & test

Out-of-tree CMake build (CMake 3.22+, C++17). `build/` is the pre-configured tree in this checkout
(`RelWithDebInfo`, `BUILD_TESTING=ON`); `build-rolling*/` are extra trees used for the rolling-contact work.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
ctest --test-dir build -R testConfiguration --output-on-failure   # one test
pre-commit run --all-files                                        # clang-format, cmake-format, black, flake8
```

Useful CMake options: `-DENABLE_FAST_TESTS=ON` (controller tests run only a few iterations),
`-DDISABLE_CONTROLLER_TESTS=ON` / `-DDISABLE_ROBOT_TESTS=ON` (skip the slow loader-based suites),
`-DBUILD_BENCHMARKS=ON`, `-DDISABLE_ROS=ON`, `-DMC_RTC_BUILD_STATIC=ON` (folds all loadable modules into the
main libraries instead of building shared plugins).

A Nix dev shell is wired through `.envrc` (`use flake .#mc-rtc`) if direnv is available.

Rolling-contact tests need `CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON` in the
environment (CTest sets these itself for the registered tests; set them by hand when running binaries
directly). See `rolling-contact-report/README.md` for the MuJoCo validation path.

Running a controller without a simulator:

```sh
./build/utils/mc_rtc_ticker -f <path/to/mc_rtc.yaml> --run-for 5 --no-sync
```

## Architecture

### Layering

`mc_rtc` (utilities: `Configuration`, `DataStore`, `Logger`, GUI state builder, dynamic loader)
→ `mc_rbdyn` (robot model: `Robot`, `Robots`, `RobotModule`, `Frame`, `Surface`, wrapping SpaceVecAlg/RBDyn)
→ `mc_solver` (QP solver + constraints) and `mc_tvm` (TVM function implementations)
→ `mc_tasks` (`MetaTask` and derived objectives)
→ `mc_control` (`MCController`, FSM, `MCGlobalController`, sample controllers)
with `mc_observers` (state estimation) and `mc_filter`, `mc_trajectory`, `mc_planning` alongside.

Headers in `include/<namespace>/` always pair with implementations in `src/<namespace>/`; keep that pairing.

### Runtime loading is the extension mechanism

Controllers, robot modules, observers, FSM states and global plugins are all shared libraries discovered at
runtime by `mc_rtc::ObjectLoader` (`MC_RTC_CONTROLLER`, `MC_RTC_ROBOT_MODULE`, `MC_RTC_GLOBAL_PLUGIN`, …
symbol prefixes). Adding one means: a new shared library, the matching registration macro in the source
(`CONTROLLER_CONSTRUCTOR`, `ROBOT_MODULE_DEFAULT_CONSTRUCTOR`, `ROBOT_MODULE_CANONIC_CONSTRUCTOR`, the
observer/plugin equivalents), and installation to the loader path. There is no central registry to edit.
CMake helpers exist for this: `add_controller` / `add_fsm_controller` (`src/mc_control/CMakeLists.txt`) and
`add_robot` (`src/mc_robots/CMakeLists.txt`).

Because modules are loaded by `dlopen`, an ABI or version mismatch shows up as a load-time failure, not a
link error — the `*_CHECK_VERSION` macros exist to catch this.

### Two QP backends: Tasks and TVM

`mc_solver::QPSolver` is abstract with `QPSolver::Backend { Tasks, TVM, Unset }`; concrete solvers are
`TasksQPSolver` and `TVMQPSolver`. This is the single most important thing to know when touching tasks or
constraints: **most tasks and constraints have two implementations**. The Tasks-side code lives in
`src/mc_solver/` and `src/mc_tasks/`; the TVM-side is a `*Function` in `src/mc_tvm/` (e.g.
`mc_tvm/TransformFunction.cpp` backs `mc_tasks::TransformTask`). A change to one backend that isn't
mirrored in the other silently degrades that backend. `mc_control::TasksController` / `TVMController` are
helper bases that pin the backend and give a typed `solver()`.

### Control loop

`MCGlobalController` does not control; it hosts `MCController` instances and drives them.
`MCGlobalController::run()` (`src/mc_control/mc_global_controller.cpp`) sequences, per tick:
global plugins `before()` → observer pipelines → `controller_->run()` (which runs `solver().run()`) →
grippers → GUI server request handling/publishing → global plugins `after()` → `logger().log()`.
Controller switching, reset and observer-pipeline teardown are handled here too.

`MCController` subclasses override `run()`, `reset()`, and optionally `createObserverPipelines()` /
`runObserverPipelines()`. Everything in `run()` is real-time context: no blocking I/O, no allocation-heavy
work, no thread creation.

`robots()` are the *control* robots (the QP's model); `realRobots()` are the estimated ones, updated by the
observer pipeline. Confusing the two is a common source of bugs.

### FSM

`mc_control::fsm::Controller` runs an `Executor` over a `TransitionMap`. States implement
`configure()` / `start()` / `run()` / `teardown()` (`include/mc_control/fsm/State.h`) and are themselves
loadable libraries, so FSM behaviour is composed in YAML rather than in C++.

### Configuration

`mc_rtc::Configuration` is the universal JSON/YAML wrapper used by every loadable component. The default
global configuration is `doc/_examples/yaml/mc_rtc/mc_rtc.yaml` (installed as the system default); user
overrides go in `~/.config/mc_rtc/mc_rtc.yaml`, and extra files can be layered via the
`MC_RTC_CONTROLLER_CONFIG` env var (colon-separated, applied last to first) or `mc_rtc_ticker -f`.
Key entries: `MainRobot`, `Enabled`, `Timestep`, `ObserverPipelines`, `Plugins`, `ControllerModulePaths`,
`RobotModulePaths`.

JSON schemas for these files live in `doc/_data/schemas/`; `.nvim.lua` wires them into `yamlls` so YAML
configs validate in-editor.

### Logging and GUI

The logger writes binary `.bin` logs; `utils/` provides `mc_bin_to_log`, `mc_bin_to_flat`, `mc_bin_utils`,
`mc_bin_perf` and the `mc_log_ui` Python GUI. The GUI is server-side state (`mc_rtc::gui::StateBuilder`)
published over a network server (`ControllerServer`) to external clients such as `mc_rtc_ros`.

### Generated sources

Files named `*.in.cpp`, `*.in.h`, `*.in.yaml`, `*.in.py`, `*.in.yml` are `configure_file` inputs — edit the
`.in.*` source, never the copy in the build tree. `3rd-party/` is vendored (qhull, RapidJSON, mpack); don't
modify it unless the dependency itself is the subject of the change.

## Rolling-contact work (local, not upstream)

An in-progress feature spanning several layers; changes here usually need edits in more than one place:

- `src/mc_rbdyn/RollingContact.cpp`, `src/mc_robots/rolling_contact.*` (+ `rolling_contact_aliases.in.yml`)
- `src/mc_solver/RollingContactConstraint.cpp`, `RollingContactDynamicsConstraint.cpp` (Tasks backend)
- `src/mc_tvm/RollingContactFunction.cpp` (TVM backend)
- `src/mc_control/samples/RollingContact/` — sample controller, its lifecycle test and CTest registrations
- `tests/testRollingContact*.cpp`
- `rolling-contact-report/` — report sources, configs, scripts, MuJoCo runner and evidence; not part of the
  upstream project

The controller tests here run `mc_rtc_ticker` against configs in `rolling-contact-report/config/` and then
validate the produced log with `rolling-contact-report/scripts/check-controller-log.py`, chained via CTest
fixtures — a failing `*Log` test usually means the run test before it produced a bad trajectory, not that
the checker is broken.
