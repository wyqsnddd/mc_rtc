# RangerMiniV3 Robot Module Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the `RollingContact` robot module — its C++ class, its source files, its CMake target, its
loader-registered name, its aliases and its description directory — to `RangerMiniV3`, and fix every dependent file
so the rename is complete.

**Architecture:** This is a pure identifier rename: no behaviour, geometry, dynamics or QP code changes. The robot
module is a `dlopen`-loaded shared library discovered by `mc_rbdyn::RobotLoader`, so its identity is spread across
four independent namespaces that must all move together — the C++ symbol, the CMake target name, the string passed
to `MC_RTC_ROBOT_MODULE` / `RobotLoader::register_object` / `_canonicalParameters`, and the alias keys in the
generated `aliases/*.yml`. Missing any one of them produces a *runtime* load failure, not a compile error.

**Tech Stack:** C++17, CMake 3.22+, mc_rtc `ObjectLoader`/`RobotLoader`, Boost.Test, CTest, YAML configuration,
mc_mujoco.

---

## The rename table

Renamed:

| Kind | Old | New |
|---|---|---|
| C++ class | `mc_robots::RollingContactRobotModule` | `mc_robots::RangerMiniV3RobotModule` |
| Header / source | `src/mc_robots/rolling_contact.{h,cpp}` | `src/mc_robots/ranger_mini_v3.{h,cpp}` |
| CMake target / `.so` | `rolling_contact` | `ranger_mini_v3` |
| Loader module name | `"RollingContact"` | `"RangerMiniV3"` |
| Alias template | `src/mc_robots/rolling_contact_aliases.in.yml` | `src/mc_robots/ranger_mini_v3_aliases.in.yml` |
| Generated alias file | `<build>/src/mc_robots/aliases/rolling_contact.yml` | `.../aliases/ranger_mini_v3.yml` |
| Alias key | `RollingContactRangerMiniV3` | `RangerMiniV3Robot` (see landmine 7) |
| Alias key | `RollingContactDifferential` | `RangerMiniV3Differential` |
| Alias key | `RollingContactFourSteering` | `RangerMiniV3FourSteering` |
| Alias key | *(new — see landmine 6)* | `RangerMiniV3Ramps` |
| Description dir | `src/mc_robots/rolling_contact_description/` | `src/mc_robots/ranger_mini_v3_description/` |
| Install path | `<datadir>/mc_rtc/rolling_contact` | `<datadir>/mc_rtc/ranger_mini_v3` |
| CMake vars | `ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH`, `ROLLING_CONTACT_INSTALL_PATH`, `ROLLING_CONTACT_DESCRIPTION_PATH` | `RANGER_MINI_V3_…` |
| Compile definitions | `ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH`, `ROLLING_CONTACT_ROBOT_MODULE_PATH` | `RANGER_MINI_V3_DESCRIPTION_SOURCE_PATH`, `RANGER_MINI_V3_ROBOT_MODULE_PATH` |

**NOT renamed** — everything below keeps the name `RollingContact` because it names the rolling-contact *feature*,
*controller* or *constraint*, not the robot module:

- The controller: `mc_control::MCRollingContactController`, its library `rolling_contact_controller`, its sources
  `src/mc_control/samples/RollingContact/mc_rolling_contact_controller.{h,cpp}`, its directory, and the controller
  name `"RollingContact"` / `"RollingContact_TVM"` used by `Enabled:` and `Default:` in every configuration.
- The controller settings section `RollingContact:` in YAML and `config("RollingContact")` in C++.
- The observer pipeline name `RollingContactPipeline`.
- `mc_rbdyn::RollingContact`, `mc_solver::RollingContactConstraint`,
  `mc_solver::RollingContactDynamicsConstraint`, `mc_tvm::RollingContactFunction` and their headers/sources.
- Test executables and CTest test names: `testRollingContactRobot`, `testRollingContactSolver`,
  `testRollingContactControllerLifecycle`, `RollingContactControllerFourSteering*`, `RollingContactCPULinkage`,
  `RollingContactRangerFrame`, and the CTest fixture names `RollingContactDifferentialLog`,
  `RollingContactFourSteeringLog`, `RollingContactFourSteeringTiltLog`.
  *Stated assumption:* these name the feature under test and are referenced verbatim by
  `doc/_i18n/en/tutorials/samples/rolling-contact.md:121` (`ctest -R 'RollingContact|testRollingContact'`) and by the
  report README. Renaming them was not requested and would break those documented commands. If the reviewer wants
  them renamed too, that is a separate follow-up.
- `ROLLING_CONTACT_OBSERVER_MODULE_PATH` and the `ROLLING_CONTACT_CHECKER` CMake variable in
  `src/mc_control/samples/RollingContact/CMakeLists.txt` — they point at the observer library and the log checker,
  not at the robot module.
- Historical records, left exactly as written: `rolling-contact-report/evidence/`,
  `rolling-contact-report/plans/` (except this file), `rolling-contact-report/*.tex`,
  `rolling-contact-report/commit-theory-correspondence.html`, `rolling-contact-report/Task-Rolling-Contact.md`,
  `rolling-contact-report/Steering-wheel-task.md`, `rolling-contact-report/rename-task.md`.

## Landmines — read before editing

1. **Substring collisions.** A blind `sed s/RollingContactFourSteering/RangerMiniV3FourSteering/` corrupts the
   CTest fixture names `RollingContactFourSteeringLog` and `RollingContactFourSteeringTiltLog`; the same applies to
   `RollingContactDifferential` vs `RollingContactDifferentialLog`. Always match the **quoted** string
   (`"RollingContactDifferential"`) in C++, or anchor with a word boundary (`RollingContactDifferential\b`).
2. **`"RollingContact"` is two different names.** In `src/mc_robots/rolling_contact.cpp`, `tests/*.cpp` and the
   first element of `MainRobot:` it is the *robot module*. In `Enabled:`, `Default:`, the `RollingContact:` YAML
   section and `config("RollingContact")` it is the *controller*. Only the first kind changes.
3. **Stale build artifacts silently keep the old name alive.** `RobotLoader` loads *every* `.so` in a module path
   and *every* `.yml` in its `aliases/` subdirectory. CMake does not delete the old `rolling_contact.so` or
   `aliases/rolling_contact.yml` when the target is renamed, so a half-cleaned build tree will still resolve
   `RollingContactRangerMiniV3` and the rename will look complete when it is not. Task 7 purges them explicitly.
4. **The installed mc_mujoco mappings hard-code the description path.** `~/local/share/mc_mujoco/ranger_mini_v3.yaml`,
   `rolling_4s.yaml`, `rolling_diff.yaml` and `ground.yaml` contain
   `src/mc_robots/rolling_contact_description/...` and, in `ground.yaml`, a `RollingContact:` section keyed on the
   first element of `MainRobot`. They must be re-copied from the repository after the rename or mc_mujoco will not
   find the MuJoCo model. Task 12 hands this over.
5. **`_canonicalParameters[0]` is the module name.** `src/mc_robots/rolling_contact.cpp:26` must change too, or
   canonical-robot loading resolves a module that no longer exists.
6. **The rename collapses two keys that `ground.yaml` relied on being different — a fourth alias restores it.**
   mc_mujoco keys its `ground` model lookup on the first element of `MainRobot`, which is the *alias key* for the
   one-argument form and the *module name* for the three-argument form. Today those differ
   (`RollingContactRangerMiniV3` vs `RollingContact`), and that difference is the only thing separating the
   flat-ground configurations from the ramp ones. After the rename both are the string `RangerMiniV3`, so a
   `RangerMiniV3:` section in `ground.yaml` would put **every** Ranger configuration on the ramps, the keyboard
   profile included. The fix is a fourth alias, `RangerMiniV3Ramps`, resolving to the same robot and used only by
   the two ramp configurations (Task 4 Step 1, Task 9 Steps 2-4). This is the one place where the rename is not
   purely mechanical; review it first.

## File structure

Moved (`git mv`, content edited afterwards):

- `src/mc_robots/rolling_contact.h` → `src/mc_robots/ranger_mini_v3.h`
- `src/mc_robots/rolling_contact.cpp` → `src/mc_robots/ranger_mini_v3.cpp`
- `src/mc_robots/rolling_contact_aliases.in.yml` → `src/mc_robots/ranger_mini_v3_aliases.in.yml`
- `src/mc_robots/rolling_contact_description/` → `src/mc_robots/ranger_mini_v3_description/`

Modified:

- CMake: `src/mc_robots/CMakeLists.txt`, `src/mc_control/samples/RollingContact/CMakeLists.txt`,
  `src/mc_control/samples/RangerTrajectory/CMakeLists.txt`, `tests/CMakeLists.txt`
- C++: `tests/testRollingContactRobot.cpp`, `tests/testRollingContactSolver.cpp`,
  `src/mc_control/samples/RollingContact/test_controller_lifecycle.cpp`,
  `src/mc_control/samples/RangerTrajectory/test_controller_lifecycle.cpp`
- Configurations: 12 files under `rolling-contact-report/config/`, 4 under `rolling-contact-report/mujoco/`
- Scripts: `rolling-contact-report/scripts/{run-mujoco-suite.py,run-ramp-validation.py,make-trajectory-waypoints.py,build-cpu-mujoco-runner.sh}`
- Docs: `CLAUDE.md`, `rolling-contact-report/{README.md,architecture.md,control-issue.md,issues.md}`,
  `doc/_i18n/{en,jp}/tutorials/samples/list-of-samples.md`,
  `rolling-contact-report/mujoco/rolling_contact_mujoco_runner.cpp` (comments only)

## Commit strategy

A module rename has no green intermediate state: the tree does not compile between the first `git mv` and the last
CMake edit, and CTest is red until the report configurations are updated. So there are exactly two commits:

- **Commit A** (after Task 9) — the whole functional rename. Tree builds; CTest expected green.
- **Commit B** (after Task 11) — documentation.

Do not commit in between.

---

### Task 1: Move the files

**Files:**
- Move: `src/mc_robots/rolling_contact.h` → `src/mc_robots/ranger_mini_v3.h`
- Move: `src/mc_robots/rolling_contact.cpp` → `src/mc_robots/ranger_mini_v3.cpp`
- Move: `src/mc_robots/rolling_contact_aliases.in.yml` → `src/mc_robots/ranger_mini_v3_aliases.in.yml`
- Move: `src/mc_robots/rolling_contact_description/` → `src/mc_robots/ranger_mini_v3_description/`

- [ ] **Step 1: Move the four paths with `git mv` so history follows them**

```bash
cd /home/yuquan/local/mc_rtc_sources/mc_rtc
git mv src/mc_robots/rolling_contact.h              src/mc_robots/ranger_mini_v3.h
git mv src/mc_robots/rolling_contact.cpp            src/mc_robots/ranger_mini_v3.cpp
git mv src/mc_robots/rolling_contact_aliases.in.yml src/mc_robots/ranger_mini_v3_aliases.in.yml
git mv src/mc_robots/rolling_contact_description    src/mc_robots/ranger_mini_v3_description
```

- [ ] **Step 2: Verify the moves are staged as renames and nothing was left behind**

Run:
```bash
git status --short src/mc_robots | head -20
ls src/mc_robots/ranger_mini_v3_description/urdf
test ! -e src/mc_robots/rolling_contact.h && test ! -e src/mc_robots/rolling_contact_description && echo OK
```
Expected: `git status` lines beginning with `R` for the header, source and alias template plus the description
files; the `urdf` listing shows `ranger_mini_v3.urdf`, `rolling_4s.urdf`, `rolling_diff.urdf`; and `OK` is printed.

Do **not** build yet — the tree is intentionally broken until Task 5.

---

### Task 2: Rewrite the robot module header

**Files:**
- Modify: `src/mc_robots/ranger_mini_v3.h`

- [ ] **Step 1: Replace the whole file**

```cpp
/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include <mc_rbdyn/RobotModule.h>

#include "api.h"

namespace mc_robots
{

/** Robot module for the Ranger Mini V3 model and the tracked rolling-contact validation robots. */
struct MC_ROBOTS_DLLAPI RangerMiniV3RobotModule : public mc_rbdyn::RobotModule
{
  RangerMiniV3RobotModule(const std::string & descriptionPath, const std::string & variant);
};

} // namespace mc_robots
```

- [ ] **Step 2: Verify no old identifier survives in the header**

Run: `grep -c "RollingContact\|rolling_contact" src/mc_robots/ranger_mini_v3.h`
Expected: `0`

---

### Task 3: Rewrite the robot module source

**Files:**
- Modify: `src/mc_robots/ranger_mini_v3.cpp`

The file keeps all of its behaviour — the variant whitelist, the `_default_attitude` z offsets, the two body
sensors, the frames and the `_stance` loop are untouched. Only the six identity sites change.

- [ ] **Step 1: Fix the include (line 5)**

Replace:
```cpp
#include "rolling_contact.h"
```
with:
```cpp
#include "ranger_mini_v3.h"
```

- [ ] **Step 2: Fix the constructor definition (line 17)**

Replace:
```cpp
RollingContactRobotModule::RollingContactRobotModule(const std::string & descriptionPath, const std::string & variant)
```
with:
```cpp
RangerMiniV3RobotModule::RangerMiniV3RobotModule(const std::string & descriptionPath, const std::string & variant)
```

- [ ] **Step 3: Fix the variant error message (line 22)**

Replace:
```cpp
    mc_rtc::log::error_and_throw<std::invalid_argument>("Unknown rolling-contact robot variant: {}", variant);
```
with:
```cpp
    mc_rtc::log::error_and_throw<std::invalid_argument>("Unknown RangerMiniV3 robot variant: {}", variant);
```

(No test asserts this message — verified against `tests/testRollingContactRobot.cpp` and
`src/mc_control/samples/RollingContact/test_controller_lifecycle.cpp`.)

- [ ] **Step 4: Fix the canonical parameters (line 26)**

Replace:
```cpp
  _canonicalParameters = {"RollingContact", descriptionPath, variant};
```
with:
```cpp
  _canonicalParameters = {"RangerMiniV3", descriptionPath, variant};
```

- [ ] **Step 5: Fix the shared-library entry points (lines 66-85)**

Replace:
```cpp
extern "C"
{
  ROBOT_MODULE_API void MC_RTC_ROBOT_MODULE(std::vector<std::string> & names)
  {
    ROBOT_MODULE_CHECK_VERSION("RollingContact")
    names = {"RollingContact"};
  }

  ROBOT_MODULE_API void destroy(mc_rbdyn::RobotModule * ptr)
  {
    delete ptr;
  }

  ROBOT_MODULE_API mc_rbdyn::RobotModule * create(const std::string &,
                                                  const std::string & descriptionPath,
                                                  const std::string & variant)
  {
    return new mc_robots::RollingContactRobotModule(descriptionPath, variant);
  }
}
```
with:
```cpp
extern "C"
{
  ROBOT_MODULE_API void MC_RTC_ROBOT_MODULE(std::vector<std::string> & names)
  {
    ROBOT_MODULE_CHECK_VERSION("RangerMiniV3")
    names = {"RangerMiniV3"};
  }

  ROBOT_MODULE_API void destroy(mc_rbdyn::RobotModule * ptr)
  {
    delete ptr;
  }

  ROBOT_MODULE_API mc_rbdyn::RobotModule * create(const std::string &,
                                                  const std::string & descriptionPath,
                                                  const std::string & variant)
  {
    return new mc_robots::RangerMiniV3RobotModule(descriptionPath, variant);
  }
}
```

- [ ] **Step 6: Fix the static (`MC_RTC_BUILD_STATIC`) registration (lines 94-101)**

Replace:
```cpp
static auto registered = []()
{
  using fn_t = std::function<mc_robots::RollingContactRobotModule *(const std::string &, const std::string &)>;
  mc_rbdyn::RobotLoader::register_object(
      "RollingContact", fn_t([](const std::string & path, const std::string & variant)
                             { return new mc_robots::RollingContactRobotModule(path, variant); }));
  return true;
}();
```
with:
```cpp
static auto registered = []()
{
  using fn_t = std::function<mc_robots::RangerMiniV3RobotModule *(const std::string &, const std::string &)>;
  mc_rbdyn::RobotLoader::register_object("RangerMiniV3",
                                         fn_t([](const std::string & path, const std::string & variant)
                                              { return new mc_robots::RangerMiniV3RobotModule(path, variant); }));
  return true;
}();
```

This block only compiles under `-DMC_RTC_BUILD_STATIC=ON`, which the default build tree does not use — it will not
be caught by the Task 7 build. Re-read it carefully.

- [ ] **Step 7: Verify no old identifier survives in the source**

Run: `grep -n "RollingContact\|rolling_contact" src/mc_robots/ranger_mini_v3.cpp`
Expected: no output.

---

### Task 4: Rewrite the alias template

**Files:**
- Modify: `src/mc_robots/ranger_mini_v3_aliases.in.yml`

`RangerMiniV3RangerMiniV3` would be absurd, so the real robot takes the bare alias and the two abstract validation
models keep descriptive suffixes. The `@RANGER_MINI_V3_DESCRIPTION_PATH@` placeholder is substituted twice by
`configure_file` in Task 5 — once with the source-tree path, once with the install path.

`RangerMiniV3Ramps` is new. It resolves to exactly the same robot as `RangerMiniV3`; its only purpose is to give the
two MuJoCo ramp configurations a distinct first `MainRobot` element for mc_mujoco's `ground.yaml` lookup to key on
(landmine 6). Without it the ramp terrain would apply to every Ranger configuration.

- [ ] **Step 1: Replace the whole file**

```yaml
RangerMiniV3Differential: [RangerMiniV3, "@RANGER_MINI_V3_DESCRIPTION_PATH@", rolling_diff]
RangerMiniV3FourSteering: [RangerMiniV3, "@RANGER_MINI_V3_DESCRIPTION_PATH@", rolling_4s]
RangerMiniV3: [RangerMiniV3, "@RANGER_MINI_V3_DESCRIPTION_PATH@", ranger_mini_v3]
# Same robot as RangerMiniV3 above. Kept as a separate key so mc_mujoco, which
# resolves the `ground` model from a section of ground.yaml keyed by the first
# element of MainRobot (mj_sim.cpp:233-248), can select the ramp terrain for the
# configurations that use this spelling and flat ground for every other one.
RangerMiniV3Ramps: [RangerMiniV3, "@RANGER_MINI_V3_DESCRIPTION_PATH@", ranger_mini_v3]
```

- [ ] **Step 2: Verify**

Run: `cat src/mc_robots/ranger_mini_v3_aliases.in.yml`
Expected: exactly the four alias lines above plus the comment, no `RollingContact` anywhere.

---

### Task 5: Rewrite `src/mc_robots/CMakeLists.txt`

**Files:**
- Modify: `src/mc_robots/CMakeLists.txt:80-110`

- [ ] **Step 1: Replace the `rolling_contact` block**

Replace lines 80-110, currently:
```cmake
add_robot(jvrc1)
add_robot(rolling_contact)

set(ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/rolling_contact_description"
)
set(ROLLING_CONTACT_INSTALL_PATH
    "${CMAKE_INSTALL_FULL_DATADIR}/mc_rtc/rolling_contact"
)
set(ROLLING_CONTACT_DESCRIPTION_PATH
    "${ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH}"
)
configure_file(
  rolling_contact_aliases.in.yml
  "${CMAKE_CURRENT_BINARY_DIR}/aliases/rolling_contact.yml"
  @ONLY
)
set(ROLLING_CONTACT_DESCRIPTION_PATH "${ROLLING_CONTACT_INSTALL_PATH}")
configure_file(
  rolling_contact_aliases.in.yml
  "${CMAKE_CURRENT_BINARY_DIR}/install-aliases/rolling_contact.yml"
  @ONLY
)
install(
  DIRECTORY rolling_contact_description/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/mc_rtc/rolling_contact"
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/install-aliases/rolling_contact.yml"
  DESTINATION "${MC_ROBOTS_RUNTIME_INSTALL_PREFIX}/aliases"
)
```
with:
```cmake
add_robot(jvrc1)
add_robot(ranger_mini_v3)

set(RANGER_MINI_V3_DESCRIPTION_SOURCE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/ranger_mini_v3_description"
)
set(RANGER_MINI_V3_INSTALL_PATH
    "${CMAKE_INSTALL_FULL_DATADIR}/mc_rtc/ranger_mini_v3"
)
set(RANGER_MINI_V3_DESCRIPTION_PATH
    "${RANGER_MINI_V3_DESCRIPTION_SOURCE_PATH}"
)
configure_file(
  ranger_mini_v3_aliases.in.yml
  "${CMAKE_CURRENT_BINARY_DIR}/aliases/ranger_mini_v3.yml"
  @ONLY
)
set(RANGER_MINI_V3_DESCRIPTION_PATH "${RANGER_MINI_V3_INSTALL_PATH}")
configure_file(
  ranger_mini_v3_aliases.in.yml
  "${CMAKE_CURRENT_BINARY_DIR}/install-aliases/ranger_mini_v3.yml"
  @ONLY
)
install(
  DIRECTORY ranger_mini_v3_description/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/mc_rtc/ranger_mini_v3"
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/install-aliases/ranger_mini_v3.yml"
  DESTINATION "${MC_ROBOTS_RUNTIME_INSTALL_PREFIX}/aliases"
)
```

The trailing `if(TARGET jvrc1) ... endif()` block at the end of the file is unrelated and stays as-is.

- [ ] **Step 2: Verify**

Run: `grep -n "rolling_contact\|ROLLING_CONTACT" src/mc_robots/CMakeLists.txt`
Expected: no output.

---

### Task 6: Update the dependent CMakeLists

**Files:**
- Modify: `src/mc_control/samples/RollingContact/CMakeLists.txt:45,46,54,115`
- Modify: `src/mc_control/samples/RangerTrajectory/CMakeLists.txt:58,68`
- Modify: `tests/CMakeLists.txt:121,122,247,249,258,260`

- [ ] **Step 1: `src/mc_control/samples/RollingContact/CMakeLists.txt` — compile definitions (lines 44-47)**

Replace:
```cmake
    PRIVATE
      ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH="${PROJECT_SOURCE_DIR}/src/mc_robots/rolling_contact_description"
      ROLLING_CONTACT_ROBOT_MODULE_PATH="$<TARGET_FILE_DIR:rolling_contact>"
      ROLLING_CONTACT_OBSERVER_MODULE_PATH="$<TARGET_FILE_DIR:EncoderObserver>"
```
with:
```cmake
    PRIVATE
      RANGER_MINI_V3_DESCRIPTION_SOURCE_PATH="${PROJECT_SOURCE_DIR}/src/mc_robots/ranger_mini_v3_description"
      RANGER_MINI_V3_ROBOT_MODULE_PATH="$<TARGET_FILE_DIR:ranger_mini_v3>"
      ROLLING_CONTACT_OBSERVER_MODULE_PATH="$<TARGET_FILE_DIR:EncoderObserver>"
```

`ROLLING_CONTACT_OBSERVER_MODULE_PATH` is deliberately unchanged — it names the observer library.
`RANGER_MINI_V3_DESCRIPTION_SOURCE_PATH` is currently unused by this test's sources; keep it defined and renamed so
the two definitions stay consistent.

- [ ] **Step 2: `src/mc_control/samples/RollingContact/CMakeLists.txt` — dependency (line 53-56)**

Replace:
```cmake
  add_dependencies(
    testRollingContactControllerLifecycle rolling_contact EncoderObserver
    BodySensorObserver VelocityAidedTiltObserver
  )
```
with:
```cmake
  add_dependencies(
    testRollingContactControllerLifecycle ranger_mini_v3 EncoderObserver
    BodySensorObserver VelocityAidedTiltObserver
  )
```

- [ ] **Step 3: `src/mc_control/samples/RollingContact/CMakeLists.txt` — CPU linkage audit (line 115)**

Replace:
```cmake
      "$<TARGET_FILE:rolling_contact>"
```
with:
```cmake
      "$<TARGET_FILE:ranger_mini_v3>"
```

The neighbouring `"$<TARGET_FILE:rolling_contact_controller>"` and `$<TARGET_FILE_DIR:rolling_contact_controller>`
lines are the *controller* library — leave them alone.

- [ ] **Step 4: `src/mc_control/samples/RangerTrajectory/CMakeLists.txt` (lines 58, 68)**

Replace:
```cmake
      RANGER_TRAJECTORY_ROBOT_MODULE_PATH="$<TARGET_FILE_DIR:rolling_contact>"
```
with:
```cmake
      RANGER_TRAJECTORY_ROBOT_MODULE_PATH="$<TARGET_FILE_DIR:ranger_mini_v3>"
```

and replace:
```cmake
  add_dependencies(
    testRangerTrajectoryControllerLifecycle rolling_contact EncoderObserver
    BodySensorObserver MetaTasks
  )
```
with:
```cmake
  add_dependencies(
    testRangerTrajectoryControllerLifecycle ranger_mini_v3 EncoderObserver
    BodySensorObserver MetaTasks
  )
```

The macro name `RANGER_TRAJECTORY_ROBOT_MODULE_PATH` is scoped to the RangerTrajectory sample and stays.

- [ ] **Step 5: `tests/CMakeLists.txt` — the `RollingContactRangerFrame` test paths (lines 121-122)**

Replace:
```cmake
      --urdf ${PROJECT_SOURCE_DIR}/src/mc_robots/rolling_contact_description/urdf/ranger_mini_v3.urdf
      --mujoco ${PROJECT_SOURCE_DIR}/src/mc_robots/rolling_contact_description/mujoco/ranger_mini_v3.xml
```
with:
```cmake
      --urdf ${PROJECT_SOURCE_DIR}/src/mc_robots/ranger_mini_v3_description/urdf/ranger_mini_v3.urdf
      --mujoco ${PROJECT_SOURCE_DIR}/src/mc_robots/ranger_mini_v3_description/mujoco/ranger_mini_v3.xml
```

The CTest name `RollingContactRangerFrame` on line 117 stays (see "NOT renamed").

- [ ] **Step 6: `tests/CMakeLists.txt` — the two robot/solver test blocks (lines 244-260)**

Replace:
```cmake
target_compile_definitions(
  testRollingContactRobot
  PRIVATE
    ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH="${PROJECT_SOURCE_DIR}/src/mc_robots/rolling_contact_description"
)
add_dependencies(testRollingContactRobot rolling_contact)
```
with:
```cmake
target_compile_definitions(
  testRollingContactRobot
  PRIVATE
    RANGER_MINI_V3_DESCRIPTION_SOURCE_PATH="${PROJECT_SOURCE_DIR}/src/mc_robots/ranger_mini_v3_description"
)
add_dependencies(testRollingContactRobot ranger_mini_v3)
```

and replace:
```cmake
target_compile_definitions(
  testRollingContactSolver
  PRIVATE
    ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH="${PROJECT_SOURCE_DIR}/src/mc_robots/rolling_contact_description"
)
add_dependencies(testRollingContactSolver rolling_contact)
```
with:
```cmake
target_compile_definitions(
  testRollingContactSolver
  PRIVATE
    RANGER_MINI_V3_DESCRIPTION_SOURCE_PATH="${PROJECT_SOURCE_DIR}/src/mc_robots/ranger_mini_v3_description"
)
add_dependencies(testRollingContactSolver ranger_mini_v3)
```

- [ ] **Step 7: Verify the CMake graph no longer names the old target**

Run:
```bash
grep -rn "rolling_contact\b\|ROLLING_CONTACT_DESCRIPTION\|ROLLING_CONTACT_ROBOT_MODULE_PATH" \
  src/mc_robots/CMakeLists.txt src/mc_control/samples/RollingContact/CMakeLists.txt \
  src/mc_control/samples/RangerTrajectory/CMakeLists.txt tests/CMakeLists.txt
```
Expected: no output. (`rolling_contact_controller` and `rolling_contact_mujoco_runner` do not match `\b` after
`rolling_contact` because the next character is `_`, a word character — if you see them, your grep dropped `\b`.)

---

### Task 7: Update the C++ sources that name the module or its aliases

**Files:**
- Modify: `tests/testRollingContactRobot.cpp:24-35,130,222`
- Modify: `tests/testRollingContactSolver.cpp:207-232`
- Modify: `src/mc_control/samples/RollingContact/test_controller_lifecycle.cpp:31` + 26 quoted alias sites
- Modify: `src/mc_control/samples/RangerTrajectory/test_controller_lifecycle.cpp:75,79`

- [ ] **Step 1: `tests/testRollingContactRobot.cpp` — the module loader helper (lines 24-35)**

Replace:
```cpp
mc_rbdyn::RobotModulePtr loadModule(const std::string & variant)
{
  configureRobotLoader();
  if(variant == "rolling_diff") { return mc_rbdyn::RobotLoader::get_robot_module("RollingContactDifferential"); }
  if(variant == "rolling_4s") { return mc_rbdyn::RobotLoader::get_robot_module("RollingContactFourSteering"); }
  if(variant == "ranger_mini_v3")
  {
    return mc_rbdyn::RobotLoader::get_robot_module("RollingContactRangerMiniV3");
  }
  return mc_rbdyn::RobotLoader::get_robot_module(
      "RollingContact", std::string(ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH), variant);
}
```
with:
```cpp
mc_rbdyn::RobotModulePtr loadModule(const std::string & variant)
{
  configureRobotLoader();
  if(variant == "rolling_diff") { return mc_rbdyn::RobotLoader::get_robot_module("RangerMiniV3Differential"); }
  if(variant == "rolling_4s") { return mc_rbdyn::RobotLoader::get_robot_module("RangerMiniV3FourSteering"); }
  if(variant == "ranger_mini_v3") { return mc_rbdyn::RobotLoader::get_robot_module("RangerMiniV3"); }
  return mc_rbdyn::RobotLoader::get_robot_module("RangerMiniV3", std::string(RANGER_MINI_V3_DESCRIPTION_SOURCE_PATH),
                                                 variant);
}
```

- [ ] **Step 2: `tests/testRollingContactRobot.cpp` — the two alias assertions (lines 130 and 222)**

Replace:
```cpp
  BOOST_CHECK_EQUAL(module->parameters()[0], "RollingContactDifferential");
```
with:
```cpp
  BOOST_CHECK_EQUAL(module->parameters()[0], "RangerMiniV3Differential");
```

and replace:
```cpp
  BOOST_CHECK_EQUAL(module->parameters()[0], "RollingContactRangerMiniV3");
```
with:
```cpp
  BOOST_CHECK_EQUAL(module->parameters()[0], "RangerMiniV3");
```

The neighbouring `canonicalParameters()[2]` assertions check the *variant* (`"rolling_diff"`, `"ranger_mini_v3"`) —
variants do not change, leave them.

- [ ] **Step 3: `tests/testRollingContactSolver.cpp` — the three loader helpers (lines 207-232)**

Replace:
```cpp
mc_rbdyn::RobotsPtr loadRollingRobots()
{
  configureRobotLoader();
  auto four = mc_rbdyn::RobotLoader::get_robot_module(
      "RollingContact", std::string(ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH), std::string("rolling_4s"));
  auto differential = mc_rbdyn::RobotLoader::get_robot_module(
      "RollingContact", std::string(ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH), std::string("rolling_diff"));
  return mc_rbdyn::loadRobots({four, differential});
}

mc_rbdyn::RobotsPtr loadDifferentialRobot()
{
  configureRobotLoader();
  auto differential = mc_rbdyn::RobotLoader::get_robot_module(
      "RollingContact", std::string(ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH), std::string("rolling_diff"));
  return mc_rbdyn::loadRobot(*differential);
}

mc_rbdyn::RobotsPtr loadFourSteeringRobot()
{
  configureRobotLoader();
  auto four = mc_rbdyn::RobotLoader::get_robot_module(
      "RollingContact", std::string(ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH), std::string("rolling_4s"));
  return mc_rbdyn::loadRobot(*four);
}
```
with:
```cpp
mc_rbdyn::RobotsPtr loadRollingRobots()
{
  configureRobotLoader();
  auto four = mc_rbdyn::RobotLoader::get_robot_module(
      "RangerMiniV3", std::string(RANGER_MINI_V3_DESCRIPTION_SOURCE_PATH), std::string("rolling_4s"));
  auto differential = mc_rbdyn::RobotLoader::get_robot_module(
      "RangerMiniV3", std::string(RANGER_MINI_V3_DESCRIPTION_SOURCE_PATH), std::string("rolling_diff"));
  return mc_rbdyn::loadRobots({four, differential});
}

mc_rbdyn::RobotsPtr loadDifferentialRobot()
{
  configureRobotLoader();
  auto differential = mc_rbdyn::RobotLoader::get_robot_module(
      "RangerMiniV3", std::string(RANGER_MINI_V3_DESCRIPTION_SOURCE_PATH), std::string("rolling_diff"));
  return mc_rbdyn::loadRobot(*differential);
}

mc_rbdyn::RobotsPtr loadFourSteeringRobot()
{
  configureRobotLoader();
  auto four = mc_rbdyn::RobotLoader::get_robot_module(
      "RangerMiniV3", std::string(RANGER_MINI_V3_DESCRIPTION_SOURCE_PATH), std::string("rolling_4s"));
  return mc_rbdyn::loadRobot(*four);
}
```

- [ ] **Step 4: `src/mc_control/samples/RollingContact/test_controller_lifecycle.cpp` — the module path macro (line 31)**

Replace:
```cpp
    mc_rbdyn::RobotLoader::update_robot_module_path({ROLLING_CONTACT_ROBOT_MODULE_PATH});
```
with:
```cpp
    mc_rbdyn::RobotLoader::update_robot_module_path({RANGER_MINI_V3_ROBOT_MODULE_PATH});
```

- [ ] **Step 5: `src/mc_control/samples/RollingContact/test_controller_lifecycle.cpp` — the 26 quoted alias sites**

These are pure literal substitutions. Match the **full quoted string** so the `RollingContact` settings key
(`config("RollingContact")`, `config.add("RollingContact")`, `invalidTerrain("RollingContact")`, …) is untouched:

```bash
sed -i \
  -e 's/"RollingContactRangerMiniV3"/"RangerMiniV3"/g' \
  -e 's/"RollingContactDifferential"/"RangerMiniV3Differential"/g' \
  -e 's/"RollingContactFourSteering"/"RangerMiniV3FourSteering"/g' \
  src/mc_control/samples/RollingContact/test_controller_lifecycle.cpp
```

- [ ] **Step 6: `src/mc_control/samples/RangerTrajectory/test_controller_lifecycle.cpp` (line 79)**

Replace:
```cpp
  return mc_rbdyn::RobotLoader::get_robot_module("RollingContactRangerMiniV3");
```
with:
```cpp
  return mc_rbdyn::RobotLoader::get_robot_module("RangerMiniV3");
```

Line 75's `RANGER_TRAJECTORY_ROBOT_MODULE_PATH` macro name is unchanged (only its CMake value moved, Task 6 Step 4).

- [ ] **Step 7: Verify the settings key survived and the aliases are gone**

Run:
```bash
grep -c 'config("RollingContact")\|config.add("RollingContact")' \
  src/mc_control/samples/RollingContact/test_controller_lifecycle.cpp
grep -rnE '"RollingContact(RangerMiniV3|Differential|FourSteering)"|ROLLING_CONTACT_(DESCRIPTION_SOURCE_PATH|ROBOT_MODULE_PATH)' \
  tests/testRollingContactRobot.cpp tests/testRollingContactSolver.cpp \
  src/mc_control/samples/RollingContact/test_controller_lifecycle.cpp \
  src/mc_control/samples/RangerTrajectory/test_controller_lifecycle.cpp
```
Expected: a non-zero count on the first command (the controller settings key is still there), and **no output**
from the second.

---

### Task 8: Purge stale artifacts and build

This is the gate for Tasks 1-7. Nothing before this point compiles.

**Files:**
- No source changes; build-tree hygiene only.

- [ ] **Step 1: Delete the stale robot module library and alias files from every build tree**

CMake does not remove artifacts of a renamed target, and `RobotLoader` loads *every* `.so` in a module path and
*every* `.yml` in its `aliases/` directory — a leftover `rolling_contact.so` plus `aliases/rolling_contact.yml`
would keep resolving `RollingContactRangerMiniV3` and mask an incomplete rename.

```bash
cd /home/yuquan/local/mc_rtc_sources/mc_rtc
for b in build build-rolling build-rolling-cpu; do
  test -d "$b" || continue
  rm -f "$b"/src/mc_robots/rolling_contact.so
  rm -f "$b"/src/mc_robots/aliases/rolling_contact.yml
  rm -f "$b"/src/mc_robots/install-aliases/rolling_contact.yml
done
```

- [ ] **Step 2: Confirm the stale files are gone**

Run: `find build build-rolling build-rolling-cpu -name 'rolling_contact.so' -o -name 'rolling_contact.yml' 2>/dev/null`
Expected: no output.

- [ ] **Step 3: Reconfigure and build**

Run:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
cmake --build build -j
```
Expected: configure and build both succeed with no errors.

- [ ] **Step 4: Confirm the renamed library and alias file were produced**

Run:
```bash
ls build/src/mc_robots/ranger_mini_v3.so build/src/mc_robots/aliases/ranger_mini_v3.yml
cat build/src/mc_robots/aliases/ranger_mini_v3.yml
```
Expected: both paths exist, and the alias file reads (with the absolute source path substituted, and the comment
from Task 4 preserved):
```yaml
RangerMiniV3Differential: [RangerMiniV3, "/home/yuquan/local/mc_rtc_sources/mc_rtc/src/mc_robots/ranger_mini_v3_description", rolling_diff]
RangerMiniV3FourSteering: [RangerMiniV3, "/home/yuquan/local/mc_rtc_sources/mc_rtc/src/mc_robots/ranger_mini_v3_description", rolling_4s]
RangerMiniV3: [RangerMiniV3, "/home/yuquan/local/mc_rtc_sources/mc_rtc/src/mc_robots/ranger_mini_v3_description", ranger_mini_v3]
RangerMiniV3Ramps: [RangerMiniV3, "/home/yuquan/local/mc_rtc_sources/mc_rtc/src/mc_robots/ranger_mini_v3_description", ranger_mini_v3]
```

- [ ] **Step 5: Confirm the shared library exports the new module name and only that**

Run:
```bash
strings build/src/mc_robots/ranger_mini_v3.so | grep -x "RangerMiniV3"
strings build/src/mc_robots/ranger_mini_v3.so | grep -x "RollingContact"
```
Expected: the first prints `RangerMiniV3`; the second prints nothing.

---

### Task 9: Update the report configurations, MuJoCo mappings and scripts

**Files:**
- Modify: `rolling-contact-report/config/mc_rtc-{four-steering,four-steering-tvm}.yaml:1`,
  `mc_rtc-four-steering-tilt.yaml:31`, `mc_rtc-ranger-mini-v3-keyboard.yaml:1`,
  `mc_rtc-ranger-tilt-ticker.yaml:37`, `mc_rtc-ranger-trajectory.yaml:1`,
  `mc_rtc-ranger-trajectory-mujoco.yaml:18`, `mc_rtc-ranger-trajectory-flat-mujoco.yaml:13`
- Modify: `rolling-contact-report/config/mc_rtc-differential{,-tvm,-modes}.yaml:1`
- Modify: `rolling-contact-report/config/mc_rtc-ranger-ramps-mujoco.yaml:14-20,44`,
  `mc_rtc-ranger-trajectory-ramps-mujoco.yaml:38`
- Modify: `rolling-contact-report/mujoco/{ground,ranger_mini_v3,rolling_4s,rolling_diff}.yaml`
- Modify: `rolling-contact-report/mujoco/rolling_contact_mujoco_runner.cpp:57,499` (comments)
- Modify: `rolling-contact-report/scripts/{run-mujoco-suite.py,run-ramp-validation.py,make-trajectory-waypoints.py,build-cpu-mujoco-runner.sh}`

- [ ] **Step 1: Rewrite the one-argument `MainRobot` lines**

Only the `MainRobot:` key changes. `Enabled:`, `Default:`, the `RollingContact:` settings section and
`RollingContactPipeline` are the controller and the observer pipeline — they stay.

```bash
cd /home/yuquan/local/mc_rtc_sources/mc_rtc/rolling-contact-report/config
sed -i 's/^MainRobot: RollingContactRangerMiniV3$/MainRobot: RangerMiniV3/' \
  mc_rtc-four-steering.yaml mc_rtc-four-steering-tvm.yaml mc_rtc-four-steering-tilt.yaml \
  mc_rtc-ranger-mini-v3-keyboard.yaml mc_rtc-ranger-tilt-ticker.yaml mc_rtc-ranger-trajectory.yaml \
  mc_rtc-ranger-trajectory-mujoco.yaml mc_rtc-ranger-trajectory-flat-mujoco.yaml
sed -i 's/^MainRobot: RollingContactDifferential$/MainRobot: RangerMiniV3Differential/' \
  mc_rtc-differential.yaml mc_rtc-differential-tvm.yaml mc_rtc-differential-modes.yaml
```

- [ ] **Step 2: Switch the two ramp configurations to the `RangerMiniV3Ramps` alias**

These two used the three-argument module form purely so that their first `MainRobot` element differed from the
one-argument configurations' — see landmine 6. The `RangerMiniV3Ramps` alias added in Task 4 now carries that
distinction, and it is shorter and does not hard-code the description path.

In `rolling-contact-report/config/mc_rtc-ranger-ramps-mujoco.yaml:44` and
`rolling-contact-report/config/mc_rtc-ranger-trajectory-ramps-mujoco.yaml:38`, replace:
```yaml
MainRobot: [RollingContact, "src/mc_robots/rolling_contact_description", ranger_mini_v3]
```
with:
```yaml
MainRobot: RangerMiniV3Ramps
```

- [ ] **Step 3: Update the ramp-selection comment block in `mc_rtc-ranger-ramps-mujoco.yaml` (lines 13-22)**

Replace:
```yaml
# THE ONE LINE THAT SELECTS THE TERRAIN
# `MainRobot` is spelled below in the three-argument module form rather than as
# the `RollingContactRangerMiniV3` alias. It resolves to exactly the same robot
# -- the alias is [RollingContact, <description path>, ranger_mini_v3]
# (src/mc_robots/rolling_contact_aliases.in.yml:3) -- but mc_mujoco keys its
# `ground` model lookup on the first element of `MainRobot`, so this form picks
# up the `RollingContact:` section of ground.yaml and with it
# src/mc_robots/rolling_contact_description/mujoco/ramp_terrain.xml. Change it
# back to the alias and the same configuration runs on flat ground. The
# description path is repo relative, so run from the repository root.
```
with:
```yaml
# THE ONE LINE THAT SELECTS THE TERRAIN
# `MainRobot` is spelled below as the `RangerMiniV3Ramps` alias rather than as
# the `RangerMiniV3` one. Both resolve to exactly the same robot
# (src/mc_robots/ranger_mini_v3_aliases.in.yml) -- but mc_mujoco keys its
# `ground` model lookup on the first element of `MainRobot`, so this spelling
# picks up the `RangerMiniV3Ramps:` section of ground.yaml and with it
# src/mc_robots/ranger_mini_v3_description/mujoco/ramp_terrain.xml. Change it
# back to `RangerMiniV3` and the same configuration runs on flat ground. The
# terrain path in ground.yaml is repo relative, so run from the repository root.
```

- [ ] **Step 4: Rewrite `rolling-contact-report/mujoco/ground.yaml`**

The `RollingContact:` key here is a robot *name* for mc_mujoco's lookup, not the controller. Replace the whole file
with:

```yaml
# mc_mujoco model mapping for the `ground` robot.
#
# Install it next to the other mappings, i.e. copy it to mc_mujoco's model
# folder (in this workspace the installed binary uses
# /home/yuquan/local/share/mc_mujoco for both its user and its share folder;
# the CPU suite's isolated build uses
# /tmp/rolling-contact-cpu-mujoco/mc_mujoco-user):
#
#   cp rolling-contact-report/mujoco/ground.yaml ~/local/share/mc_mujoco/
#
# WHY THIS FILE EXISTS
# mc_control::MCController always loads `env/ground` as a second robot
# (src/mc_control/MCController.cpp:96) and mc_mujoco resolves one MuJoCo model
# per mc_rtc robot from `<module name>.yaml` (mj_sim.cpp:110-127). `ground.yaml`
# is therefore the hook that decides what the world looks like, without editing
# the robot description and without touching the controller.
#
# HOW THE SELECTION WORKS -- READ THIS BEFORE EDITING
# mc_mujoco prefers a section keyed by the FIRST element of `MainRobot` over the
# top-level `xmlModelPath` (mj_sim.cpp:233-248).
#     MainRobot: RangerMiniV3
# yields the key "RangerMiniV3", which is absent here, so those configurations
# fall through to the top-level entry and keep the stock flat ground -- the
# existing suite, the keyboard profile and every unrelated robot are unaffected
# by installing this file.
#     MainRobot: RangerMiniV3Ramps
# yields the key "RangerMiniV3Ramps" and selects the ramp terrain. The two
# aliases resolve to exactly the same robot
# (src/mc_robots/ranger_mini_v3_aliases.in.yml); the spelling is the only
# difference between config/mc_rtc-ranger-mini-v3-keyboard.yaml and
# config/mc_rtc-ranger-ramps-mujoco.yaml.
#
# The default below is an absolute path because it has to keep working for
# runs started from any directory; the ramp entry is repo relative because it
# only ever applies to this repository's own configurations, which the README
# already requires to be launched from the repository root.
xmlModelPath: /home/yuquan/local/share/mc_mujoco/env/ground.xml
RangerMiniV3Ramps:
  xmlModelPath: src/mc_robots/ranger_mini_v3_description/mujoco/ramp_terrain.xml
```

- [ ] **Step 5: Update the three per-robot MuJoCo mappings**

```bash
cd /home/yuquan/local/mc_rtc_sources/mc_rtc
sed -i 's|src/mc_robots/rolling_contact_description|src/mc_robots/ranger_mini_v3_description|g' \
  rolling-contact-report/mujoco/ranger_mini_v3.yaml \
  rolling-contact-report/mujoco/rolling_4s.yaml \
  rolling-contact-report/mujoco/rolling_diff.yaml \
  rolling-contact-report/mujoco/rolling_contact_mujoco_runner.cpp
```

These three file *names* (`ranger_mini_v3.yaml`, `rolling_4s.yaml`, `rolling_diff.yaml`) are keyed on
`RobotModule::name`, which is the **variant** and does not change. Do not rename them.

- [ ] **Step 6: Update the scripts**

```bash
cd /home/yuquan/local/mc_rtc_sources/mc_rtc/rolling-contact-report/scripts
sed -i -e 's|src/mc_robots/rolling_contact_description|src/mc_robots/ranger_mini_v3_description|g' \
       -e 's/"RollingContactDifferential"/"RangerMiniV3Differential"/g' \
       -e 's/"RollingContactRangerMiniV3"/"RangerMiniV3"/g' \
       run-mujoco-suite.py
sed -i -e 's|src/mc_robots/rolling_contact_description|src/mc_robots/ranger_mini_v3_description|g' \
       -e 's/^MainRobot: RollingContactRangerMiniV3$/MainRobot: RangerMiniV3/' \
       run-ramp-validation.py
sed -i 's|${MC_RTC_BUILD}/src/mc_robots/rolling_contact.so|${MC_RTC_BUILD}/src/mc_robots/ranger_mini_v3.so|' \
  build-cpu-mujoco-runner.sh
```

`run-mujoco-suite.py:473` and `run-ramp-validation.py:243` set
`controller = "RollingContact" if backend == "Tasks" else "RollingContact_TVM"` — that is the *controller* name and
is untouched by the sed patterns above (they only match the longer quoted alias strings). Verify this after running.

Neither script needs the `RangerMiniV3Ramps` alias: both `install_terrain()` helpers write a **top-level**
`xmlModelPath:` into their own `ground.yaml` and delete the file to go back to flat ground, so they never key on the
first element of `MainRobot`. Landmine 6 does not apply to them.

- [ ] **Step 7: Update the `make-trajectory-waypoints.py` comment (lines 72-74)**

Replace:
```python
# Chassis frame height of the Ranger Mini V3 at reset: RollingContactRobotModule
# sets _default_attitude z = 0.16 for the ranger_mini_v3 variant
# (src/mc_robots/rolling_contact.cpp) and the "chassis" frame is the floating
```
with:
```python
# Chassis frame height of the Ranger Mini V3 at reset: RangerMiniV3RobotModule
# sets _default_attitude z = 0.16 for the ranger_mini_v3 variant
# (src/mc_robots/ranger_mini_v3.cpp) and the "chassis" frame is the floating
```

- [ ] **Step 8: Verify the controller name survived everywhere it should**

Run:
```bash
cd /home/yuquan/local/mc_rtc_sources/mc_rtc
grep -c "^Enabled: \[RollingContact\]\|^Default: RollingContact$" rolling-contact-report/config/*.yaml | grep -v ':0' | wc -l
grep -rn '"RollingContact"' rolling-contact-report/scripts/run-mujoco-suite.py rolling-contact-report/scripts/run-ramp-validation.py
```
Expected: a non-zero count on the first (the controller selection is intact), and the two
`controller = "RollingContact" if backend == "Tasks" ...` lines printed by the second.

- [ ] **Step 9: Verify no robot-module reference is left in configs, mappings or scripts**

Run:
```bash
grep -rnE 'RollingContact(RangerMiniV3|Differential|FourSteering)\b|rolling_contact_description|MainRobot: \[RollingContact' \
  rolling-contact-report/config rolling-contact-report/mujoco rolling-contact-report/scripts
```
Expected: no output.

- [ ] **Step 10: Verify the flat/ramp split survived the alias collapse**

Run:
```bash
cd /home/yuquan/local/mc_rtc_sources/mc_rtc
grep -rn '^MainRobot:' rolling-contact-report/config/*.yaml
grep -n '^RangerMiniV3' rolling-contact-report/mujoco/ground.yaml
```
Expected: exactly two configurations — `mc_rtc-ranger-ramps-mujoco.yaml` and
`mc_rtc-ranger-trajectory-ramps-mujoco.yaml` — read `MainRobot: RangerMiniV3Ramps`; the eight Ranger configurations
read `MainRobot: RangerMiniV3`; the three differential ones read `MainRobot: RangerMiniV3Differential`; and
`ground.yaml` has exactly one keyed section, `RangerMiniV3Ramps:`. If `RangerMiniV3:` appears as a key in
`ground.yaml`, the keyboard profile will run on the ramps — fix it before continuing.

---

### Task 10: Commit the functional rename

**Files:**
- No new edits.

- [ ] **Step 1: Run the repository formatters**

Run: `pre-commit run --all-files`
Expected: clang-format, cmake-format, black and flake8 all pass (they may reformat the wrapped lines added in
Tasks 3 and 7 — that is fine; stage the result).

- [ ] **Step 2: Rebuild after formatting**

Run: `cmake --build build -j`
Expected: success.

- [ ] **Step 3: Commit**

```bash
cd /home/yuquan/local/mc_rtc_sources/mc_rtc
git add -A src/mc_robots src/mc_control/samples/RollingContact src/mc_control/samples/RangerTrajectory \
  tests rolling-contact-report/config rolling-contact-report/mujoco rolling-contact-report/scripts
git commit -m "$(cat <<'EOF'
refactor(robots): rename the RollingContact robot module to RangerMiniV3

The module loads the Ranger Mini V3 and its two abstract validation
variants; "RollingContact" named the contact model, not the robot, and
collided with the controller and the constraint classes of the same name.

Renames the class, its sources, the CMake target, the loader-registered
module name, the description directory and the three aliases
(RangerMiniV3, RangerMiniV3Differential, RangerMiniV3FourSteering), and
updates every dependent CMakeLists, test, report configuration and
script. The RollingContact controller, constraints and observer pipeline
keep their names.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Update the documentation

**Files:**
- Modify: `CLAUDE.md:123`
- Modify: `rolling-contact-report/README.md:6,72,283,325,413,432-433`
- Modify: `rolling-contact-report/architecture.md:36`
- Modify: `rolling-contact-report/control-issue.md:135-136,213,606`
- Modify: `rolling-contact-report/issues.md:10-11`
- Modify: `doc/_i18n/en/tutorials/samples/list-of-samples.md:198,201`
- Modify: `doc/_i18n/jp/tutorials/samples/list-of-samples.md:202` (and the matching "supported robots" line)

- [ ] **Step 1: `CLAUDE.md` (line 123)**

Replace:
```markdown
- `src/mc_rbdyn/RollingContact.cpp`, `src/mc_robots/rolling_contact.*` (+ `rolling_contact_aliases.in.yml`)
```
with:
```markdown
- `src/mc_rbdyn/RollingContact.cpp`, `src/mc_robots/ranger_mini_v3.*` (+ `ranger_mini_v3_aliases.in.yml`)
```

- [ ] **Step 2: `rolling-contact-report/README.md` — the alias mentions**

```bash
cd /home/yuquan/local/mc_rtc_sources/mc_rtc
sed -i -e 's/RollingContactRangerMiniV3/RangerMiniV3/g' \
       -e 's/`RollingContactDifferential` alias/`RangerMiniV3Differential` alias/' \
       -e 's|src/mc_robots/rolling_contact_description|src/mc_robots/ranger_mini_v3_description|g' \
  rolling-contact-report/README.md
```

Note line 72 becomes `robot:=RangerMiniV3 controller:=RollingContact --run-for 5 --no-sync` — `controller:=` is the
controller and must stay `RollingContact`. Confirm that after the sed.

- [ ] **Step 3: `rolling-contact-report/README.md:430-435` — the ramp-selection paragraph**

The sed in Step 2 leaves this paragraph describing the three-argument form, which Task 9 Step 4 replaced with the
`RangerMiniV3Ramps` alias. Replace:
```markdown
The selection is one line. mc_mujoco prefers a section of `ground.yaml` keyed by the first element of `MainRobot`
over the top-level `xmlModelPath` (`mj_sim.cpp:233-248`), so `MainRobot: RangerMiniV3` keeps the stock
flat plane while `MainRobot: [RangerMiniV3, "src/mc_robots/ranger_mini_v3_description", ranger_mini_v3]` - the
same robot, spelled as the three-argument module - selects the ramps. Installing the mapping therefore changes
nothing for the existing configurations, the suite, or any other robot.
```
with:
```markdown
The selection is one line. mc_mujoco prefers a section of `ground.yaml` keyed by the first element of `MainRobot`
over the top-level `xmlModelPath` (`mj_sim.cpp:233-248`), so `MainRobot: RangerMiniV3` keeps the stock flat plane
while `MainRobot: RangerMiniV3Ramps` - an alias for exactly the same robot - selects the ramps. Installing the
mapping therefore changes nothing for the existing configurations, the suite, or any other robot.
```

- [ ] **Step 4: `rolling-contact-report/architecture.md`, `control-issue.md`, `issues.md`**

```bash
cd /home/yuquan/local/mc_rtc_sources/mc_rtc
sed -i -e 's/RollingContactRangerMiniV3/RangerMiniV3/g' \
       -e 's/MainRobot: RollingContactDifferential/MainRobot: RangerMiniV3Differential/g' \
       -e 's|src/mc_robots/rolling_contact_description|src/mc_robots/ranger_mini_v3_description|g' \
  rolling-contact-report/architecture.md rolling-contact-report/control-issue.md rolling-contact-report/issues.md
```

This fixes the two documented preflight guards
(`grep -qx 'MainRobot: RangerMiniV3' "$KEYBOARD_CFG"` in `README.md:325` and `control-issue.md:213`) so they keep
matching the configuration file.

- [ ] **Step 5: `doc/_i18n/en/tutorials/samples/list-of-samples.md` (lines 198 and 201)**

Replace:
```markdown
**Supported robots**: the included `rolling_diff` and `rolling_4s` robot modules.

```yaml
MainRobot: [RollingContact, /path/to/mc_rtc/src/mc_robots/rolling_contact_description, rolling_diff]
Enabled: RollingContact
```
```
with:
```markdown
**Supported robots**: the `rolling_diff`, `rolling_4s` and `ranger_mini_v3` variants of the included
`RangerMiniV3` robot module.

```yaml
MainRobot: [RangerMiniV3, /path/to/mc_rtc/src/mc_robots/ranger_mini_v3_description, rolling_diff]
Enabled: RollingContact
```
```

`Enabled: RollingContact` is the controller and stays.

- [ ] **Step 6: `doc/_i18n/jp/tutorials/samples/list-of-samples.md` (line 202 and the preceding robots line)**

Apply the same two substitutions to the Japanese page: the `MainRobot:` line becomes
```yaml
MainRobot: [RangerMiniV3, /path/to/mc_rtc/src/mc_robots/ranger_mini_v3_description, rolling_diff]
```
and mention `ranger_mini_v3` alongside `rolling_diff` / `rolling_4s` in the supported-robots sentence, keeping the
surrounding Japanese prose. Leave `Enabled: RollingContact` unchanged.

- [ ] **Step 7: Commit**

```bash
cd /home/yuquan/local/mc_rtc_sources/mc_rtc
pre-commit run --all-files
git add -A CLAUDE.md doc/_i18n rolling-contact-report/README.md rolling-contact-report/architecture.md \
  rolling-contact-report/control-issue.md rolling-contact-report/issues.md
git commit -m "$(cat <<'EOF'
docs(rolling): document the RangerMiniV3 robot module rename

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Final completeness gate and validation handoff

**Files:**
- No edits.

- [ ] **Step 1: Run the completeness grep**

The word boundaries are load-bearing: they exclude the CTest fixture names `RollingContactDifferentialLog`,
`RollingContactFourSteeringLog` and `RollingContactFourSteeringTiltLog`, which stay.

```bash
cd /home/yuquan/local/mc_rtc_sources/mc_rtc
grep -rnE 'RollingContactRobotModule|RollingContactRangerMiniV3|RollingContact(Differential|FourSteering)\b|rolling_contact_description|ROLLING_CONTACT_(DESCRIPTION|INSTALL|ROBOT_MODULE)|rolling_contact\.(h|cpp|so|yml)|rolling_contact_aliases' \
  --exclude-dir=build --exclude-dir=build-rolling --exclude-dir=build-rolling-cpu --exclude-dir=.git \
  --exclude-dir=evidence --exclude-dir=plans \
  --exclude='*.tex' --exclude='*.html' \
  --exclude='rename-task.md' --exclude='Task-Rolling-Contact.md' --exclude='Steering-wheel-task.md' \
  .
```
Expected: **no output**. Anything printed is an incomplete rename — fix it and re-run before continuing.

- [ ] **Step 2: Confirm the deliberately-kept names are still present**

```bash
cd /home/yuquan/local/mc_rtc_sources/mc_rtc
grep -rn "MCRollingContactController" src/mc_control/samples/RollingContact/mc_rolling_contact_controller.h | head -1
grep -rn "RollingContactFourSteeringTiltLog" src/mc_control/samples/RollingContact/CMakeLists.txt | head -1
grep -rn "RollingContactPipeline" rolling-contact-report/config/mc_rtc-ranger-mini-v3-keyboard.yaml | head -1
```
Expected: one hit each. If any is missing, an over-eager substitution damaged the controller side.

- [ ] **Step 3: Clean-tree rebuild**

Run:
```bash
rm -rf /tmp/mc-rtc-rename-verify
cmake -S . -B /tmp/mc-rtc-rename-verify -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
cmake --build /tmp/mc-rtc-rename-verify -j
```
Expected: success. A clean tree proves the rename does not depend on stale artifacts in `build/`.

- [ ] **Step 4: Hand the validation over**

Validation is the reviewer's. Report the following, verbatim, as what still needs to be run — and do **not** claim
the rename is validated until they have run it:

1. The test suite (the rolling-contact tests need the two environment variables; CTest sets them for its own
   registered tests, so the plain invocation is enough):
   ```sh
   ctest --test-dir build --output-on-failure
   ```
   The tests that exercise this rename are `testRollingContactRobot`, `testRollingContactSolver`,
   `RollingContactRangerFrame`, `RollingContactControllerRepeatedLifecycle`, `RangerTrajectoryControllerLifecycle`
   and the `RollingContactController*` ticker/log pairs.

2. **Refresh the installed mc_mujoco mappings before running the simulator.** The copies in
   `~/local/share/mc_mujoco/` still contain `src/mc_robots/rolling_contact_description/...` and a stale
   `RollingContact:` section, and mc_mujoco reads them, not the repository:
   ```sh
   cp rolling-contact-report/mujoco/ground.yaml          ~/local/share/mc_mujoco/
   cp rolling-contact-report/mujoco/ranger_mini_v3.yaml  ~/local/share/mc_mujoco/
   cp rolling-contact-report/mujoco/rolling_4s.yaml      ~/local/share/mc_mujoco/
   cp rolling-contact-report/mujoco/rolling_diff.yaml    ~/local/share/mc_mujoco/
   ```
   The stale `~/local/share/mc_mujoco/rolling_contact/` directory holds `rolling_4s.xml` / `rolling_diff.xml` and
   is not referenced by any mapping after this change; leaving or deleting it is the reviewer's call.

3. The acceptance run from the task, launched **from the repository root** (every path in the configuration is
   repo-relative):
   ```sh
   mc_mujoco -f rolling-contact-report/config/mc_rtc-ranger-mini-v3-keyboard.yaml
   ```
   Expected: the Ranger Mini V3 loads (not the JVRC1 fallback), the GUI comes up, WASD-style keys drive the
   chassis, and `x` exits — identical to before the rename. `No loadable robot with module RangerMiniV3` means the
   process was not started from the repository root, or `build/src/mc_robots` still lacks `ranger_mini_v3.so`.

---

## Self-review notes

- **Spec coverage.** `rename-task.md` asks for (a) the rename, (b) completeness across dependent files, (c)
  validation via `mc_mujoco -f rolling-contact-report/config/mc_rtc-ranger-mini-v3-keyboard.yaml`. (a) is Tasks
  1-5; (b) is Tasks 6-9 and 11, gated by Task 12 Step 1; (c) is Task 12 Step 4, handed to the reviewer per their
  instruction.
- **Design tension worth recording.** The module builds three variants — `rolling_diff`, `rolling_4s` and
  `ranger_mini_v3` — so `RangerMiniV3` names the module after one of them. This was raised and the naming was
  confirmed; `RangerMiniV3Differential` / `RangerMiniV3FourSteering` keep the other two discoverable. If the
  validation variants are ever dropped, the name becomes exactly right.
- **The one non-mechanical part.** Collapsing `RollingContactRangerMiniV3` (alias) and `RollingContact` (module)
  into the single string `RangerMiniV3` destroys the key that `ground.yaml` used to tell flat-ground from ramp
  configurations. A fourth alias, `RangerMiniV3Ramps`, restores the distinction: added in Task 4 Step 1, adopted by
  the two ramp configurations in Task 9 Step 2, keyed in `ground.yaml` in Task 9 Step 4, checked in Task 9 Step 10,
  and documented in Task 9 Step 3 and Task 11 Step 3. It adds one alias rather than changing any behaviour, but it
  is the part of this plan most worth reviewing.
- **Type/name consistency.** The strings `RangerMiniV3`, `RangerMiniV3Differential`, `RangerMiniV3FourSteering`,
  `RangerMiniV3Ramps`, `RangerMiniV3RobotModule`, target `ranger_mini_v3`, macros
  `RANGER_MINI_V3_DESCRIPTION_SOURCE_PATH` / `RANGER_MINI_V3_ROBOT_MODULE_PATH` and directory
  `ranger_mini_v3_description` are used identically in every task above; the Task 12 grep is the backstop.

## Execution outcome (2026-09-14)

Executed and validated. Two corrections were needed against the plan as written:

1. **Landmine 7 — an alias may not shadow a library-declared module.** `RobotLoader::load_aliases`
   (`src/mc_rbdyn/RobotLoader.cpp:53-57`) discards any alias whose key matches a module the loader already
   declares, and the module now registers as `RangerMiniV3`. The planned bare `RangerMiniV3:` alias was therefore
   dropped at load time, so `MainRobot: RangerMiniV3` resolved to the library module itself — whose `create()`
   requires descriptionPath + variant — and segfaulted in `testRollingContactRobot`'s
   `LoadRangerMiniV3RollingRobot`. Resolution: the real robot's alias is **`RangerMiniV3Robot`**; the module keeps
   the bare `RangerMiniV3`. Same class of collision as landmine 6, and the reason the *old* scheme worked at all —
   every old alias differed from the old module name `RollingContact`.
2. `run-ramp-validation.py:249` writes `MainRobot:` inside an f-string, so the `^MainRobot:`-anchored sed in
   Task 9 Step 6 did not match it. Edited directly.

**Validation.** `109/109` CTest tests pass with `MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON CUDA_VISIBLE_DEVICES=-1`,
including all 19 rolling-contact/Ranger tests. Without those variables 58 tests fail on a pre-existing qhull
convex-generation crash (`QH6028 ... Wrong qh_fprintf was called`) in JVRC1-based suites that CTest does not set
the variables for; that failure predates this work — the affected files are untouched by it — and the same tests
pass once the variables are set.

**Not run:** `pre-commit` and its formatters (`clang-format`, `cmake-format`, `black`, `flake8`) are not installed
in this environment. Formatting was checked by hand instead: no line over 120 columns was introduced (the 8 + 2
over-long lines in `testRollingContactSolver.cpp` and `test_controller_lifecycle.cpp` are present at HEAD), no
tabs, and the edits preserve the surrounding structure. Run `pre-commit run --all-files` on a machine that has it
before submitting.
