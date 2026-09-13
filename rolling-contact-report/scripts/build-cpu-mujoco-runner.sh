#!/bin/sh

set -eu

MC_MUJOCO_REVISION=9397c73fcc348f4f2a3d2f433dcaf948a67532ea
MUJOCO_VERSION=3.3.6
MUJOCO_SHA256=049204172901afad251070385a6badf46d795ebe47403d093f8469557eeeab5a

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)
WORK_ROOT=${ROLLING_CONTACT_MUJOCO_WORK_ROOT:-/tmp/rolling-contact-cpu-mujoco}
MC_RTC_BUILD=${ROLLING_CONTACT_MC_RTC_BUILD:-${REPOSITORY_ROOT}/build}
MC_MUJOCO_SOURCE=${WORK_ROOT}/mc_mujoco
MC_MUJOCO_BUILD=${WORK_ROOT}/mc_mujoco-build
MC_MUJOCO_PREFIX=${WORK_ROOT}/mc_mujoco-install
MUJOCO_ARCHIVE=${WORK_ROOT}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz
MUJOCO_ROOT=${WORK_ROOT}/mujoco-${MUJOCO_VERSION}
RUNNER_BUILD=${WORK_ROOT}/runner-build
USER_CONFIGURATION=${WORK_ROOT}/mc_mujoco-user
EIGENPY_SHIM=${REPOSITORY_ROOT}/rolling-contact-report/cmake/eigenpy-header-only-shim
TASKS_DIR=${MC_RTC_BUILD}/deps/tasks-system-install/lib/cmake/Tasks

if [ ! -f "${TASKS_DIR}/TasksConfig.cmake" ]; then
  echo "Tasks build-tree package is missing; build mc_rtc first: ${TASKS_DIR}" >&2
  exit 1
fi

mkdir -p "${WORK_ROOT}"

if [ ! -d "${MC_MUJOCO_SOURCE}/.git" ]; then
  git clone --recursive https://github.com/rohanpsingh/mc_mujoco.git "${MC_MUJOCO_SOURCE}"
fi

if [ "$(git -C "${MC_MUJOCO_SOURCE}" rev-parse HEAD)" != "${MC_MUJOCO_REVISION}" ]; then
  if [ -n "$(git -C "${MC_MUJOCO_SOURCE}" status --porcelain)" ]; then
    echo "mc_mujoco source has local changes; refusing to change revision" >&2
    exit 1
  fi
  git -C "${MC_MUJOCO_SOURCE}" fetch origin "${MC_MUJOCO_REVISION}"
  git -C "${MC_MUJOCO_SOURCE}" checkout --detach "${MC_MUJOCO_REVISION}"
  git -C "${MC_MUJOCO_SOURCE}" submodule update --init --recursive
fi

PATCH=${REPOSITORY_ROOT}/rolling-contact-report/patches/mc_mujoco-user-destination.patch
if git -C "${MC_MUJOCO_SOURCE}" apply --check "${PATCH}" 2>/dev/null; then
  git -C "${MC_MUJOCO_SOURCE}" apply "${PATCH}"
elif ! git -C "${MC_MUJOCO_SOURCE}" apply --reverse --check "${PATCH}" 2>/dev/null; then
  echo "mc_mujoco patch does not apply to the pinned revision" >&2
  exit 1
fi

if [ ! -f "${MUJOCO_ARCHIVE}" ]; then
  curl --fail --location --output "${MUJOCO_ARCHIVE}" \
    "https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz"
fi
echo "${MUJOCO_SHA256}  ${MUJOCO_ARCHIVE}" | sha256sum --check --status
if [ ! -f "${MUJOCO_ROOT}/include/mujoco/mujoco.h" ]; then
  tar -xzf "${MUJOCO_ARCHIVE}" -C "${WORK_ROOT}"
fi

cmake -S "${MC_MUJOCO_SOURCE}" -B "${MC_MUJOCO_BUILD}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF \
  -DCMAKE_INSTALL_PREFIX="${MC_MUJOCO_PREFIX}" \
  -DCMAKE_PREFIX_PATH="${EIGENPY_SHIM}" \
  -Deigenpy_DIR="${EIGENPY_SHIM}/lib/cmake/eigenpy" \
  -DMC_MUJOCO_USER_DESTINATION="${USER_CONFIGURATION}" \
  -DMUJOCO_ROOT_DIR="${MUJOCO_ROOT}" \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DTasks_DIR="${TASKS_DIR}" \
  -DBoost_DIR=/usr/lib/x86_64-linux-gnu/cmake/Boost-1.83.0 \
  -DBoost_INCLUDE_DIR=/usr/include \
  -DUSE_GL=ON
cmake --build "${MC_MUJOCO_BUILD}" --parallel 2
cmake --install "${MC_MUJOCO_BUILD}"

# mc_mujoco resolves a robot module named, for example, ``ranger_mini_v3`` through
# a same-named user configuration file. These mappings are part of this report
# rather than upstream mc_mujoco, so install them explicitly in the isolated
# destination selected above.
mkdir -p "${USER_CONFIGURATION}"
cmake -E copy_if_different \
  "${REPOSITORY_ROOT}/rolling-contact-report/mujoco/rolling_diff.yaml" \
  "${USER_CONFIGURATION}/rolling_diff.yaml"
cmake -E copy_if_different \
  "${REPOSITORY_ROOT}/rolling-contact-report/mujoco/ranger_mini_v3.yaml" \
  "${USER_CONFIGURATION}/ranger_mini_v3.yaml"

cmake -S "${REPOSITORY_ROOT}/rolling-contact-report/mujoco" -B "${RUNNER_BUILD}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF \
  -DCMAKE_PREFIX_PATH="${MC_MUJOCO_PREFIX};${EIGENPY_SHIM}" \
  -Deigenpy_DIR="${EIGENPY_SHIM}/lib/cmake/eigenpy" \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DTasks_DIR="${TASKS_DIR}" \
  -DBoost_DIR=/usr/lib/x86_64-linux-gnu/cmake/Boost-1.83.0 \
  -DBoost_INCLUDE_DIR=/usr/include
cmake --build "${RUNNER_BUILD}" --parallel 2

python3 "${REPOSITORY_ROOT}/rolling-contact-report/scripts/audit-cpu-linkage.py" \
  --library-dir "${MC_RTC_BUILD}/src" \
  --library-dir "${MC_RTC_BUILD}/plugins/ROS" \
  --library-dir "${MC_RTC_BUILD}/deps/tasks-system-install/lib" \
  --output "${WORK_ROOT}/cpu-linkage.json" \
  "${RUNNER_BUILD}/rolling_contact_mujoco_runner" \
  "${MC_RTC_BUILD}/src/libmc_rbdyn.so" \
  "${MC_RTC_BUILD}/src/libmc_solver.so" \
  "${MC_RTC_BUILD}/src/mc_control/samples/RollingContact/rolling_contact_controller.so" \
  "${MC_RTC_BUILD}/src/mc_robots/ranger_mini_v3.so"

echo "CPU MuJoCo runner: ${RUNNER_BUILD}/rolling_contact_mujoco_runner"
echo "CPU linkage report: ${WORK_ROOT}/cpu-linkage.json"
