# Batched GPU migration for DynamicStability and Isaac Lab

## Assessment of the original task

The original task is directionally useful, but it is not yet a proper implementation task. It names GPU versions of
the two executables without identifying the reusable computation, its input/output contract, the numerical oracle, or
the acceptance criteria.

In this repository:

- `AreaCalculator` is a CPU command-line front end. The reusable calculation is `StandingStabilityCalc`, which calls
  `McContact`, `McZMPArea`, and `McComArea`.
- `puppet` is a DART/OSG/ImGui teleoperation and visualization application. Its DART inverse kinematics, windowing, and
  rendering are not needed by an Isaac Lab reward or observation term.
- DART is only one of the CPU dependencies in the calculation. DART provides robot parsing, forward kinematics, link
  transforms, CoM data, Jacobians, mass/bias terms, limits, and state storage. Stabiliplus/GLPK performs the polytope
  projection, while mc_rtc, GeometricRobotics, RoboticsUtils, Eigen, YAML, and Qhull/GEOS also appear in the current
  target graph.
- `training/isaaclab/g1_stability_task.py` already runs a learned TorchScript polygon surrogate on the Isaac Lab device.
  That is a useful deployment path, but it is not an exact GPU implementation of the analytical LP calculation. The
  task must state which result is wanted.

The recommended scope is therefore an **exact, batched, device-resident implementation of the analytical stability
calculation**, with the existing CPU implementation retained as the authoritative oracle during migration. Porting or
removing the GUI is not a prerequisite.

## Goal

Provide a batched API that consumes Isaac Lab robot/contact tensors on a CUDA device and returns ZMP and CoM-velocity
stability polygons, validity flags, and solver diagnostics on the same device. The steady-state Isaac Lab path must not
perform per-environment Python loops, file I/O, DART calls, or device-to-host transfers.

ZMP and CoM-velocity polygons are the first production milestone because they share the deterministic fixed-direction
`McZMPArea` implementation and already have paired use in the training pipeline. `AreaCalculator` also computes the
CoM-position polygon through `McComArea`; a backend must port it through the same gates before claiming complete
`AreaCalculator` equivalence. `McDCMArea` is not called by `StandingStabilityCalc` or `AreaCalculator` and is outside
the first migration unless an Isaac Lab consumer is identified.

The implementation must preserve the explicitly frozen CPU mathematical contract. Intentional fixes or model changes
must be evaluated separately; they must not be hidden inside a backend migration.

## Repository implementation checkpoint (2026-08-30)

This document is the migration specification and gate list, not a statement that the migration is complete. The
following foundation is implemented in this worktree:

- `LPProblemSnapshot` captures owned pre-solver inputs, contact primitives, and compact CPU `F`, `f`, `A`, and `b` for
  ZMP, CoM velocity, and CoM position. `LPProblemSnapshot_test` independently reconstructs every matrix block for each
  individual contact slot, one through four contacts, friction 0/0.3/0.7/1.0, deterministic nonzero velocity and
  acceleration, and mixed/all-disabled contact torque-limit flags. It also proves exactly which `f` blocks change
  under acceleration and which Jacobian blocks change under per-contact torque-limit flags. The CPU implementation
  now treats zero contacts as an explicit invalid assembly for all three areas: it clears every polygon and compact
  `F`, `f`, `A`, and `b`, records `assembly_valid = false` with `invalid_reason = "no active contacts"`, and recovers
  on the next valid contact update. The stateless GPU API has the matching clean invalid-result regression.
- The Python binding now converts polygon and LP inputs through explicit strided float64 buffers. This avoids the
  implicit `std::vector<Eigen::Vector2d>` converter crash under the CPU validation environment's NumPy 2.x while
  retaining non-contiguous-array support and deterministic owned inputs.
- `CUDA-task/export_cpu_oracle.py` exports checksummed, versioned, labeled artifacts. Strict nominal two-foot/64-
  direction artifacts cover every loadable repository robot: G1, GR1T2, H1, JVRC1, N1, and Talos. Each direction
  retains the complete CPU decision vector, direct-GLPK status, production-wrapper outcome, and `F*x-f`/`A*x-b`
  residuals. Export fails if the two CPU solver paths disagree for a strict solver-oracle case.
- `CUDA-task/golden/g1_matrix_variants.json` is a separate eight-case matrix-migration corpus covering every contact
  slot, one through four contacts, three friction values, nonzero `qdot`/`qdd`, and mixed/all-disabled torque limits.
  It deliberately records, rather than conceals, existing CPU solver inconsistencies: a hand-only direct solve can be
  infeasible while the legacy wrapper reports success, and some multi-contact production/direct optima differ. Those
  cases are authoritative for pre-solver matrix and feasible-witness parity, but are marked `solver_oracle_eligible =
  false`; the six nominal artifacts remain the strict solver oracle. The high-level CPU projection success flag is
  retained, but its unstable support function is intentionally omitted from these matrix-only cases.
- `CUDA-task/validate_cpu_oracle_replay.py` regenerates every artifact in five fresh processes and requires exact byte
  identity; the current seven-artifact corpus passes all 35 regenerations. Raw Stabiliplus/Qhull vertex lists are not
  retained because replay proved that the legacy projector can nondeterministically keep or drop redundant/interior
  optima. Strict ZMP and CoM-velocity cases instead retain the high-level projection's support function over the 64
  frozen directions plus its success flag and error against
  direct GLPK. The legacy `McComArea` support function is also unstable and is not retained. The canonical direct-GLPK
  polygon, direction solutions, supports, residuals, and all `F`, `f`, `A`, `b` coefficients are unrounded.
- `training/stability_gpu/assembly.py` implements the 16-plane contact cone, contact transforms, motion term, canonical
  padding/masks, and batched Torch assembly for all three area kinds. It uses no per-environment loop or host transfer.
- `training/tests/test_gpu_lp_assembly.py` compares the CPU-generated and Torch-generated `F`, `f`, `A`, and `b`
  element by element for all six robots' nominal cases and all eight G1 matrix variants in float64 and float32. It
  also tests logical labels and masks, projection modes and torque-limit semantics, checks retained direct-GLPK
  solutions against the Torch-generated matrices, and checks mixed-batch failure containment.
- `CUDA-task/validate_gpu_assembly.py` emits the Stage 2D machine-readable report, including worst row/column labels,
  absolute/relative/L2 errors, sign differences, and structural-zero mismatches.
- `training/stability_gpu/solver.py` is a pure-Torch batched predictor-corrector LP solver with deterministic invalid,
  infeasible, unbounded, numerical-failure, and max-iteration statuses. Analytical LPs, objective conventions, row
  scaling/permutation/redundancy, padding, and failure containment are covered by `test_gpu_lp_solver.py`. The strict
  per-robot reports pass all 1,152 nominal directions (six robots times three formulations times 64 directions). The
  float64 maxima are `5.26e-9` support error, `3.12e-10` equality residual, `3.72e-10` inequality violation,
  `1.23e-10` dual residual, and `9.94e-10` gap. Float32 storage with the default same-device FP64 compute policy has
  maxima `2.89e-5`, `1.25e-5`, `1.49e-5`, `6.15e-7`, and `4.96e-5`, respectively. Native float32 remains an
  experimental audit mode because its degenerate axis-aligned objectives do not all satisfy the dual/KKT certificate;
  the promoted policy is an explicit accuracy/performance variant, not an implicit tolerance relaxation.
- `training/stability_gpu/api.py` composes assembly and solving for ZMP, CoM velocity, and CoM position. It checks
  finite/convex/counter-clockwise polygon output, unique support points, area, residuals, and propagates a deterministic
  all-zero polygon sentinel if any directional solve for an environment fails. The composed float64 supports match all
  three frozen GLPK projections. `CUDA-task/run_gpu_area_calculator.py` is the thin batch-one diagnostic client over
  this same API; it loads one checksummed case and reports all three polygons, validity, statuses, support error, and
  solver residuals without duplicating assembly or solver code.
- The CPU oracle now freezes 49 generalized DOF names, 43 articulated DOF names, 66 body names, and all 66 DART body
  masses, world/local CoM positions, and inertia tensors. The loader validates their dimensions, symmetry,
  positive-semidefiniteness, and proves the per-body mass-weighted reduction reproduces the whole-robot mass and CoM
  used by every LP area.
  `CUDA-task/audit_gpu_topology.py` proves that the repository G1 URDF contains the same name sets, while the current
  29-joint learning/stock-locomotion schema is missing all 14 hand joints. The URDF and oracle orders differ, so the
  runtime must map by name. `training/isaaclab/exact_g1_asset.py` is the candidate full-URDF configuration and disables
  fixed-joint merging.
- The static inertia audit identifies 12 links without a URDF `<inertial>` block. DART assigns each one a 1 kg imported
  default, so the frozen total is 48.168234 kg while explicit URDF masses total 36.168234 kg.
  `training/stability_gpu/urdf_parity.py` materializes those frozen DART defaults into a deterministic, fully explicit
  temporary URDF and verifies that every already-explicit source inertia still agrees with the oracle. The exact Isaac
  asset uses this file; `audit_gpu_topology.py --require-exact-ready` rejects the source mass contract but proves the
  materialized asset is statically ready for live parity validation.
  `CUDA-task/audit_newton_dynamics.py` makes the resulting second-model drift executable: Newton 1.0.0 preserves the
  66-body/43-joint topology but imports 36.168232 kg, fails nominal mass-matrix parity, and has no public full
  bias/inverse-dynamics API. Newton is rejected as the dynamics source unless these facts change and all primitive
  gates are rerun.
- `training/stability_gpu/topology.py`, `primitives.py`, and `isaaclab_adapter.py` make the simulator boundary explicit.
  Tests cover reordered joint/body tensors, world-frame PhysX Jacobians, root-column order, both mass-matrix axes,
  whole-robot CoM, motion-term construction, SPD/symmetry/torque-limit validity, contact pose/cone transforms, and mixed
  batches. The simulator boundary now distinguishes PhysX `[linear, angular]`, raw DART `[angular, linear]`, and the
  frozen non-SVA `[linear, linear]` Jacobian alias described below. A complete injected dynamics triple remains an
  explicit oracle boundary, but partial triples and zero substitution are rejected.
- The production adapter no longer depends on a missing public full-bias API. `training/stability_gpu/dynamics.py`
  reduces public per-body CoM accelerations, velocities, link-frame inertias, rotations, and world Jacobians into the
  complete generalized inverse-dynamics vector `M*qdd+bias`; the LP consumes its articulated slice directly. On the
  deterministic nonzero 49-DOF CPU state, an independent Newton-Euler reconstruction agrees with DART to
  `5.69e-14`. The same module reproduces the repository's legacy momentum-like helper—including its link-origin
  spatial-inertia/CoM-velocity semantics—to `6.67e-16`. Float64/float32 oracle tests, a hand-computable gyroscopic body,
  invalid rotations/inertias, reordered bodies/joints, and mixed-batch containment cover this boundary.
- `test_gpu_isaaclab_oracle_roundtrip.py` inverts the frozen DART primitives into deliberately reversed PhysX-style
  joint/body storage, including all 66 body Jacobians, poses, CoMs, masses, inertias, velocities, and accelerations.
  It passes them through the production public-body adapter (without injected bias/acceleration/momentum vectors),
  reconstructs the nominal DART generalized gravity vector, and reproduces every active CPU-generated `F`, `f`, `A`,
  and `b` entry for ZMP, CoM velocity, and CoM position. This is the requested intermediate matrix gate; it isolates
  simulator primitive mapping plus assembly before the solver, but it does not replace a live simulator parity run.
  `IsaacLabExactStabilityBackend` then exercises the complete single-call adapter→assembly→solver path and matches all
  three frozen 64-direction support functions.
- `CUDA-task/audit_cpu_robot_coverage.py` discovers repository robot configurations rather than using a hard-coded
  allowlist. G1, GR1T2, H1, JVRC1, N1, and Talos all load, resolve all four configured contact bodies, produce finite
  correctly shaped CPU `F`, `f`, `A`, and `b` snapshots for every area, and complete the nominal projections. For each
  robot, the same audit disables every contact, requires failed update status, empty polygons, empty invalid snapshots,
  and complete state metadata, then restores the nominal contacts and requires successful assembly and projection.
  Their generalized-DOF/body counts are respectively 49/66, 38/45, 25/33, 50/68, 29/37, and 50/68.

Live CUDA/Isaac validation has now been run on the target RTX 5090 with PyTorch 2.7.0+cu128, Isaac Sim 5.1, and
Isaac Lab 2.3. The materialized 66-body/43-joint asset passes topology, body mass/local-CoM/world-CoM/inertia,
mass-matrix, contact-Jacobian, inverse-dynamics, legacy-momentum, and all three CPU-generated `F`, `f`, `A`, and `b`
matrix comparisons. Five checked fixture poses and six joint-limit/seeded-random/root-translation/root-yaw poses pass
the same live mechanics and matrix reports. The independent finite-difference gate covers all six floating-root
columns and 12 seeded articulated columns selected from the 29 columns that influence a configured contact; its worst
absolute error is `6.95e-5` with `epsilon=0.002` and `atol=rtol=2e-4`. A live mixed batch of four states passes
single-state equivalence and permutation equivariance for all 47 reported tensors with exact zero difference.

`CUDA-task/validate_isaaclab_trace.py` adds the synchronized Stage 5B gate. It replays both-feet, left-foot,
zero-contact, foot-plus-hand, hand-only-infeasible, right-foot, and both-feet-recovery frames through one live exact G1
articulation and compares both ZMP and CoM-velocity results with the matching offline CPU/GLPK cases. All statuses,
validity flags, direction-success flags, and 64-direction supports pass; the worst support error is `1.91e-5` against
the `5e-4` float32 gate, the hand-only case is classified infeasible, and invalid-to-valid recovery is bit exact. The
trace deliberately injects a deterministic ordered contact mask while retaining live simulator mechanics, so physical
contact-sensor transitions remain covered by the reward smoke rather than being misrepresented as CPU-replayable.

The exact manager task is selectable through `training/isaaclab/train_g1.py --ablation exact`. Its steady-state reward
reads the public PhysX contact view directly to avoid Isaac Lab's lazy sensor-update `nonzero()` synchronization,
keeps constants and name mappings preloaded, returns a deterministic zero fallback for invalid environments, and
publishes device-resident valid/failure/max-iteration fractions under `env.extras["log"]`. A profiled two-environment
run with all 17 rewards enabled passes finite-value and shape checks, finds no D2H or local-scalar event, holds
allocated and reserved memory flat across three measured steps, and obtains 128/128 successful exact directions.

The current gate status is therefore:

| Stage | Status | Evidence |
| --- | --- | --- |
| 0 — CPU contract/oracle | Pass | Four native CTests and 293/293 CPU-environment Python tests pass; seven checked artifacts replay byte-identically 35/35 times. |
| 1 — live robot primitives | Pass | Nominal, five fixtures, six boundary/root poses, independent FD, and B=4 equivariance pass on live PhysX. |
| 2 — tensor assembly | Pass | CPU/GPU `F`, `f`, `A`, `b`, labels, masks, residual, permutation, and metamorphic gates pass in float64/float32. |
| 3 — GPU solver | Pass for the frozen corpus | All 1,152 nominal robot/area/direction cases pass with the promoted-float32 policy; failure semantics have analytical regressions. |
| 4 — composed backend | Pass for the frozen and live conformance corpora | ZMP, CoM velocity, and CoM position are implemented; mixed batches and contact modes pass. |
| 5 — Isaac Lab integration | Pass | Live mapping, seven-frame synchronized trace, exact-reward smoke, zero-D2H profile, bounded memory, and deterministic fallback pass. |
| 6 — performance/memory | **Fail** | The exact reward consumes 90.1% of a B=64 step and 95.9% of the configured B=4096 step; the requirement is less than 10%. |
| 7 — cleanup/packaging | Blocked by Stage 6 | The specification explicitly permits this only after conformance and performance pass. CPU-only build/test compatibility is already preserved. |

The Stage 6 failure is measured rather than inferred. With the production reward settings (64 directions,
same-device float64 promotion, 20 fixed Newton iterations, direction chunks of 16), the RTX 5090 results are:

| Live batch | Exact reward | Full Isaac step | Fraction | Peak allocated | Reserved after warm-up | Solver result at sampled state |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 239.26 ms | 265.41 ms | 90.15% | 155.2 MB | 186.6 MB | 256/4,096 directions optimal; 3,840 hit the iteration limit |
| 4,096 | 9,873.60 ms | 10,296.26 ms | 95.90% | 9.16 GB | 10.56 GB | 256/262,144 directions optimal at the sampled dynamic state |

The B=64 timing uses five warm-up steps and three CUDA-event measurements; allocation and reservation are flat after
warm-up. The production-count timing is a one-sample safety run because each step already takes more than ten seconds.
The low valid count is not a contact-indexing error: the B=64 status histogram is 256 optimal and 3,840
max-iteration directions. The reward remains finite through its documented fallback and exposes this failure rate, but
that does not satisfy the Stage 6 numerical/failure acceptance gate.

Two solver optimizations were retained because they preserve the exact contract: affine and corrector steps share one
LU factorization per Newton iteration, and a fixed direction-chunk option bounds memory with bit-exact outputs. On a
repeated nominal B=64 component case, ten iterations reduce one ZMP solve to 39.61 ms and the three-area API to
122.26 ms. A saturated unchunked B=1024 case still takes 702.73 ms for one area and 1,822.13 ms for all three while
peaking at 8.34 GB. Chunking avoids a 4096-environment OOM but increases launch/factorization overhead. The following
alternatives were audited and rejected rather than silently weakening the task:

- native float32 does not certify every degenerate cardinal objective under the frozen tolerances;
- reducing 64 directions to 8/16/32 produces support errors up to `0.59` on the expanded boundary-pose corpus;
- a numerically accurate fused Triton 32-by-32 partial-pivot LU was slower than cuSOLVER (`2.629 ms` versus
  `0.635 ms` for 4,096 systems);
- cuOpt's documented batch-LP path is deprecated, while its supported Python/server model and result APIs are
  host-oriented and target much larger sparse programs, so it cannot satisfy this reward's zero-host-work contract.

Consequently the exact migration is correctness-complete through Stage 5 but is **not complete under this document's
own exit criteria**. Finishing it requires a materially different GPU projection algorithm or an explicitly approved
accuracy/performance variant; changing direction count, precision, tolerances, or evaluation frequency without rerunning
conformance would violate the frozen contract.

Current CPU/oracle and CUDA unit-validation commands are:

```sh
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=python:training \
  conda run --no-capture-output -n robot_stability_training python -m pytest -q \
  -p no:cacheprovider training/tests
for robot in g1 gr1t2 h1 jvrc1 n1 talos; do
  PYTHONPATH=python:training conda run -n robot_stability_training python \
    CUDA-task/export_cpu_oracle.py \
    --config "robot_description/${robot}/config/robot_configuration.yaml" \
    --case-set nominal --output "CUDA-task/golden/${robot}_nominal_two_foot.json" \
    --direction-count 64
done
PYTHONPATH=python:training conda run -n robot_stability_training python \
  CUDA-task/export_cpu_oracle.py --case-set matrix-variants \
  --output CUDA-task/golden/g1_matrix_variants.json --direction-count 64
PYTHONPATH=python:training conda run -n robot_stability_training python \
  CUDA-task/validate_cpu_oracle_replay.py --repetitions 5 \
  --output /tmp/cpu_oracle_replay.json
PYTHONPATH=python conda run -n robot_stability_training python \
  CUDA-task/audit_cpu_robot_coverage.py --direction-count 8 \
  --output /tmp/cpu_robot_coverage.json
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run --no-capture-output -n env_isaaclab python -m pytest -q \
  training/tests/test_cpu_oracle.py \
  training/tests/test_gpu_lp_assembly.py \
  training/tests/test_gpu_lp_solver.py \
  training/tests/test_gpu_topology.py \
  training/tests/test_gpu_dynamics.py \
  training/tests/test_gpu_primitives.py \
  training/tests/test_gpu_isaaclab_adapter.py \
  training/tests/test_gpu_isaaclab_oracle_roundtrip.py \
  training/tests/test_gpu_topology_audit.py \
  training/tests/test_gpu_urdf_parity.py \
  training/tests/test_gpu_stability_api.py
conda run --no-capture-output -n env_isaaclab python \
  CUDA-task/validate_gpu_assembly.py --device auto --dtype float64
conda run --no-capture-output -n env_isaaclab python \
  CUDA-task/validate_gpu_assembly.py --device auto --dtype float32
conda run --no-capture-output -n env_isaaclab python \
  CUDA-task/run_gpu_area_calculator.py --device auto --dtype float64 \
  --output /tmp/gpu_area_calculator.json
conda run --no-capture-output -n env_isaaclab python \
  CUDA-task/validate_gpu_assembly.py \
  --oracle CUDA-task/golden/g1_matrix_variants.json \
  --device auto --dtype float64 --output /tmp/g1_matrix_variants_float64.json
conda run --no-capture-output -n env_isaaclab python \
  CUDA-task/validate_gpu_assembly.py \
  --oracle CUDA-task/golden/g1_matrix_variants.json \
  --device auto --dtype float32 --output /tmp/g1_matrix_variants_float32.json
for robot in g1 gr1t2 h1 jvrc1 n1 talos; do
  for dtype in float64 float32; do
    conda run --no-capture-output -n env_isaaclab python \
      CUDA-task/validate_gpu_solver.py \
      --oracle "CUDA-task/golden/${robot}_nominal_two_foot.json" \
      --device auto --dtype "${dtype}" \
      --output "/tmp/${robot}_solver_${dtype}.json"
  done
done
conda run --no-capture-output -n env_isaaclab python \
  CUDA-task/audit_gpu_topology.py --require-exact-ready
WARP_CACHE_PATH=/tmp/dynamic-stability-warp-cache \
  conda run -n env_isaaclab python CUDA-task/audit_newton_dynamics.py --device cpu
```

Live Isaac conformance, trace, smoke, and timing commands are:

```sh
PYTHONPATH=python:training conda run -n robot_stability_training python \
  CUDA-task/export_cpu_oracle.py --case-set g1-fixtures --direction-count 8 \
  --output /tmp/g1_checked_fixtures_8.json
PYTHONPATH=python:training conda run -n robot_stability_training python \
  CUDA-task/export_cpu_oracle.py --case-set g1-boundary-poses --direction-count 8 \
  --output /tmp/g1_boundary_poses_8.json
conda run --no-capture-output -n env_isaaclab python \
  CUDA-task/validate_isaaclab_topology.py --headless --device cuda:0 \
  --force-usd-conversion --output /tmp/isaaclab_topology.json
conda run --no-capture-output -n env_isaaclab python \
  CUDA-task/validate_isaaclab_pose_corpus.py --headless --device cuda:0 \
  --oracle /tmp/g1_checked_fixtures_8.json --output /tmp/isaaclab_fixtures.json
conda run --no-capture-output -n env_isaaclab python \
  CUDA-task/validate_isaaclab_pose_corpus.py --headless --device cuda:0 \
  --oracle /tmp/g1_boundary_poses_8.json --output /tmp/isaaclab_boundary_poses.json
conda run --no-capture-output -n env_isaaclab python \
  CUDA-task/validate_isaaclab_jacobian_fd.py --headless --device cuda:0 \
  --joint-count 12 --output /tmp/isaaclab_jacobian_fd.json
conda run --no-capture-output -n env_isaaclab python \
  CUDA-task/validate_isaaclab_batch_equivariance.py --headless --device cuda:0 \
  --oracle /tmp/g1_boundary_poses_8.json --batch-size 4 \
  --output /tmp/isaaclab_batch_equivariance.json
conda run --no-capture-output -n env_isaaclab python \
  CUDA-task/validate_isaaclab_trace.py --headless --device cuda:0 \
  --output /tmp/isaaclab_trace.json
conda run --no-capture-output -n env_isaaclab python \
  CUDA-task/validate_isaaclab_exact_reward_smoke.py --headless --device cuda:0 \
  --num-envs 2 --warmup-steps 2 --steps 3 --timing-repetitions 3 \
  --output /tmp/isaaclab_exact_smoke.json
```

The Stage 6 component benchmark uses CUDA events and one synchronization after each measurement series. The live smoke
validator measures complete reward and simulator-step time separately. Reproduce the retained nominal component result
and the failing production-count gate with:

```sh
conda run --no-capture-output -n env_isaaclab python \
  CUDA-task/benchmark_gpu_stability.py --device cuda --dtype float32 \
  --batch-sizes 1,64,1024 --max-iterations 10 --warmup 5 --repetitions 20 \
  --output /tmp/gpu_stability_benchmark.json
conda run --no-capture-output -n env_isaaclab python \
  CUDA-task/validate_isaaclab_exact_reward_smoke.py --headless --device cuda:0 \
  --num-envs 4096 --warmup-steps 1 --steps 1 --timing-repetitions 1 \
  --skip-profiler --skip-clean-shutdown --output /tmp/isaaclab_exact_b4096.json
```

The 4096-environment command intentionally skips the profiler because the two-environment profiled smoke already proves
the host-transfer gate and retaining a production-sized Torch trace consumes unnecessary memory. It still measures the
complete live reward and full Isaac step with CUDA events. Its expected exit status is zero for smoke correctness and
bounded allocation, while `performance_gate_less_than_ten_percent` is false; Stage 6 remains failed.

## Non-goals for the first migration

- Reimplementing the `puppet` GUI, DART inverse kinematics, OSG rendering, or ImGui on the GPU.
- Deleting the CPU implementation or its dependencies before GPU conformance is demonstrated.
- Requiring gradients through the LP solver. The first Isaac Lab integration is an inference/reward calculation unless
  a separate use case establishes a need for differentiability.
- Claiming that the existing learned TorchScript surrogate is numerically equivalent to the analytical solver.
- Expanding the contact model beyond the current oracle while establishing backend parity.
- Claiming full `AreaCalculator` parity before its CoM-position output is also implemented and validated.

## Current CPU contract that must be frozen

Before implementing kernels, document these behaviors with tests or an oracle export. Several are easy to change
accidentally during a rewrite:

- The state vector passed to `StandingStabilityCalc` contains the six floating-base degrees of freedom followed by the
  articulated degrees of freedom. Isaac Lab represents root and joint state differently, so an explicit mapping and
  quaternion convention are required.
- The constraint construction uses link poses, CoM position and height, world Jacobians, total mass, the mass matrix,
  Coriolis/gravity forces, joint accelerations, joint velocities, force limits, and contact parameters.
- `StandingStabilityCalc::updateRobotState` sets positions and velocities but not accelerations. Nevertheless,
  `McZMPArea::constructProblem_` reads DART accelerations. The migration must either add acceleration to the public
  contract or freeze it to zero and test that assumption.
- The `comLinearVelocities` argument is currently ignored. The “CoM-velocity area” is created by changing the projection
  matrix using `omega = sqrt(g / com_z)` and `pTwo = -0.2`; it is not translated by the supplied CoM velocity. Preserve
  this behavior for parity or approve a separately tested semantic change.
- The contact wrench cone is a 16-plane inner approximation for a rectangular patch. The implementation uses
  `mu / sqrt(2)`, `2 * halfX`, and `2 * halfY`, with different force/moment layouts controlled by
  `useSpatialVectorAlgebra`.
- In the current non-SVA path, `McContact::computeJacobian` does not complete its intended DART
  `[angular, linear]` to `[linear, angular]` row swap. `auto rotJacMat = jac.block(...)` aliases the top block; after
  the first assignment, the second copies the new value and produces `[linear, linear]`. The checked-in oracle and
  torque constraints freeze this behavior. The exact adapter reproduces it explicitly; correcting it requires a
  separately reviewed CPU semantic change and regenerated oracle, not an unnoticed migration cleanup.
- The current configuration accepts only local `+Z` contact normals. `minForce` is parsed but is not applied in the LP.
- In `McZMPArea`, `applyJointTorqueLimit` controls whether each contact Jacobian contributes to articulated torque
  constraints. The floating-base rows are excluded using `aDof = dof - 6`. `McComArea` currently applies those blocks
  unconditionally, so complete `AreaCalculator` parity must preserve or separately correct that difference.
- Contact and friction overrides mutate calculator state. Passing an empty contact map reuses the last active set, and a
  later friction value of `-1` does not restore the original YAML values. Oracle generation should use a fresh
  calculator per independent sample, or this statefulness must be reproduced deliberately.
- On projection failure, the CPU implementation reports failure but retains vertices from the previous successful
  update. A batched GPU API should instead return an explicit validity mask and deterministic invalid values; consumers
  must never treat stale vertices as a current solution.
- Deterministic CPU regression uses 64 equally spaced LP directions. The five G1 fixtures then convert the result to 24
  ordered vertices in `AreaCalculator_test`.

## Proposed runtime contract

Use a stateless tensor API. Exact names may change, but the contract must make all state visible:

```text
inputs (all batched, leading dimension B)
  root_pose_world             [B, 7]
  root_spatial_velocity       [B, 6]
  joint_position              [B, J]
  joint_velocity              [B, J]
  joint_acceleration          [B, J] or explicitly fixed to zero
  active_contacts             [B, C] bool
  friction                    [B, C]
  zmp_height                  [B]
  robot/contact constants     preloaded once on the device

outputs
  zmp_vertices                [B, D, 2]
  com_velocity_vertices       [B, D, 2]
  valid                       [B, 2] bool
  unique_vertex_count         [B, 2]
  max_equality_residual       [B, 2]
  max_inequality_violation    [B, 2]
  solver_status               [B, 2]
```

`C` is a fixed ordered contact-slot schema, and `D` is a fixed ordered direction count, initially 64. Fixed shapes avoid
host-side ragged lists and are suitable for thousands of parallel environments. Duplicate directional optima may remain
in the runtime representation; polygon cleanup or conversion to 24 rays should be a separate batched postprocessor.

The API must document coordinate frames, wrench ordering, joint ordering, units, dtypes, clockwise/counter-clockwise
ordering, and the invalid-output sentinel. Configuration/YAML parsing may remain on the CPU during one-time startup;
“GPU implementation” means the per-step numerical path is device resident, not that every dependency is a CUDA library.

## Migration stages and intermediate validations

Do not start the next stage until the current exit gate passes. Each comparison must report the maximum error and the
identity of the worst sample, rather than only returning a pass/fail value.

### Stage 0 — Freeze scope, semantics, and the CPU baseline

1. Decide explicitly between:
   - exact analytical GPU solver (the scope recommended here),
   - the existing learned TorchScript surrogate, or
   - both, exposed as distinct backends with distinct accuracy claims.
2. Keep `StandingStabilityCalc` and the CPU tests as the oracle.
3. Run the existing deterministic tests and save toolchain/dependency versions.
4. Add regression cases for:
   - zero, one, two, three, and four active contacts, including a clean invalid result for zero contacts;
   - per-contact torque-limit on/off;
   - at least two friction values;
   - nonzero root/joint velocity and the agreed acceleration behavior;
   - projection failure without reuse of stale output by the new API;
   - all repository robots that successfully load, not only G1.

Current baseline commands:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

**Validation 0A — deterministic replay:** run every oracle sample five times in a fresh process. Success flags, matrix
dimensions, active-contact ordering, and 64-direction support values must be identical. Existing G1 24-vertex fixtures
must continue to pass at their current `1e-9` ordered-vertex tolerance.

**Validation 0B — oracle corpus:** add an export utility that records inputs and the following intermediates in a
versioned, deterministic format:

- link/contact transforms, CoM, Jacobians, mass matrix, bias vector, and force limits;
- local and inertial contact-wrench-cone matrices and resultant-wrench multipliers;
- compact CPU LP matrices/vectors `F`, `f`, `A`, and `b` for ZMP, CoM velocity, and CoM position, before the solver
  scales or reorders them;
- canonical fixed-contact-slot/padded forms of those matrices, plus row/column labels and active masks;
- per-direction objective, solution vertex, solver status, and constraint residuals;
- final validity flags and polygons.

Expose these values through a typed, read-only test snapshot taken immediately after problem construction and before
solver preprocessing. Do not scrape formatted debug output, and do not export a matrix after GLPK or Stabiliplus has
scaled, permuted, or mutated it. The snapshot must own its data so a later direction solve cannot change an earlier
artifact.

Include the robot/config hashes, dependency revisions, direction count, dtype, coordinate conventions, and generator
command. Large generated corpora should be CI artifacts rather than committed binaries; commit only compact golden
cases.

**Validation 0C — oracle-export self-consistency:** solve the exported `F,f,A,b` matrices through both the production
Stabiliplus wrapper and an independently loaded direct GLPK path for the same 64 directions. Retain each direct solve's
complete decision vector and verify `F*x <= f`, `A*x = b`, solver status, support, output point, and reconstructed
polygon before the artifact can be used as a GPU oracle. Deliberately corrupt one stored coefficient, dimension,
contact permutation, checksum, and primal witness in separate negative tests and prove that the loader rejects each
one. Keep GLPK preprocessing settings identical between the two paths; a different preprocessing mode is a different
numerical oracle for these poorly scaled dynamics matrices and must be evaluated separately.

**Exit:** the CPU suite is green, the ambiguities above are resolved in writing, and a small deterministic golden corpus
can be regenerated byte-for-byte or field-for-field and passes its self-consistency test.

### Stage 1 — Prove the robot-data backend before choosing cuRobo

Do not assume cuRobo is a drop-in DART replacement. First create a capability matrix for each quantity used by
`DartRobotHelpers.hpp` and `McZMPArea::constructProblem_`. For every quantity, identify whether it comes directly from
Isaac Lab/PhysX tensors, cuRobo, a custom batched implementation, or a precomputed robot constant.

The spike must cover:

- URDF joint/link ordering and fixed transforms;
- floating-base pose convention;
- world link/contact poses;
- CoM position, total mass, and CoM height;
- 6-by-DOF world Jacobians with the same angular/linear row convention as DART;
- mass matrix, Coriolis/gravity or equivalent bias term, and acceleration convention;
- effort limits and their mapping to articulated joints.

Use cuRobo only if the spike demonstrates the required quantities, conventions, supported robot descriptions, batch
behavior, and compatible license/version. If Isaac Lab already exposes a required tensor, prefer that state source to a
second robot model that could drift from the simulator.

The locally installed Isaac Lab 0.47.2 source gives the following initial capability matrix. “Available” here means an
API exists; it does not mean its frame or numerical convention already matches DART.

| Required quantity | Candidate source | Current gate/status |
| --- | --- | --- |
| Joint/body names and ordering | `Articulation.joint_names`, `body_names` | Available; explicit G1 permutation still required |
| Root and body link pose | `ArticulationData.root_link_*`, `body_link_pose_w` | Available in world frame, quaternion `wxyz`; parity not run |
| Per-body CoM pose/velocity/acceleration | `body_com_pose_w`, `body_com_vel_w`, `body_com_acc_w` | Public world-frame `[linear, angular]` tensors; offline DART reconstruction passes, live non-nominal parity pending |
| Body mass/inertia | `get_masses()`, `get_inertias()` | Inertia is about the CoM in link actor frame; static DART-default parity asset and offline elementwise tests pass, live import parity pending |
| Whole-robot CoM and mass | Batched mass-weighted reduction over body CoMs | Implemented and checked against all 66 CPU bodies |
| World contact Jacobian | `root_physx_view.get_jacobians()` | Isaac Lab treats the direct tensor as world-frame `[linear, angular]`; body-point and floating-base column parity still require finite-difference proof |
| Generalized mass matrix | `root_physx_view.get_generalized_mass_matrices()` | Available; floating-base column order and dimensions require proof |
| Complete inverse dynamics | Batched Newton-Euler reduction of public body tensors, `sum(J_com^T*wrench)` | Implemented without a separate robot model; nonzero DART oracle agrees to `5.69e-14`, live non-nominal parity pending |
| Gravity/Coriolis split | Not required by the LP production path | The LP consumes `(M*qdd+bias)[6:]`; the injected oracle boundary retains separate vectors for diagnostic tests |
| Generalized acceleration | Encoded by public per-body CoM accelerations in the production reduction | Avoids finite-differenced joint/root acceleration assembly; live sampling semantics remain a gate |
| Effort limits | `ArticulationData.joint_effort_limits` | Available; map to DART articulated slice and verify actuator-vs-URDF values |
| Frozen CPU `lDot` approximation | Pure-Torch reproduction of `getSimulatedCentroidalMomentumDMoment` | Implemented with exact legacy semantics; nonzero CPU oracle agrees to `6.67e-16`, live parity pending |
| Contact state/force | Isaac Lab `ContactSensor` | Available; semantic slot and threshold policy must be explicit |

Newton 1.0.0 was also evaluated as the installed device-side dynamics alternative. Its public API provides a
floating-base Jacobian and generalized mass matrix, but not a full bias/inverse-dynamics result. More importantly, its
import of the same repository URDF assigns zero mass to the 12 links without explicit inertials, whereas DART assigned
1 kg to each. The audited Newton total is 36.168232 kg versus 48.168234 kg in the CPU oracle, and its nominal mass
matrix consequently fails parity. This is a concrete example of why a second model cannot be accepted from topology
and API availability alone.

This capability analysis selects the public per-body PhysX reduction instead of cuRobo/Newton for the current backend.
That selection is provisional until floating-base Jacobians, body accelerations, and the reconstructed inverse-dynamics
and legacy-momentum quantities pass the live Validation 1B–1E gates.

**Validation 1A — topology/config:** for every supported robot, compare joint/link names, parent indices, actuated DOF
counts, effort limits, masses, inertial parameters, and contact body resolution. Fail on a missing or duplicate mapping;
never silently fill a mechanics input with a training mean.

**Validation 1B — primitive parity:** compare at least the reference pose, the five checked-in G1 fixture poses, joint
limit poses, seeded random valid poses, and translated/rotated floating-base poses. Validate transforms, CoM, Jacobians,
mass matrix, bias forces, and `omega` independently. A failure here must not be debugged through final polygon output.

**Validation 1C — Jacobian finite differences:** for every contact body and a seeded subset of joints, perturb one
coordinate at a time and compare finite-difference world translation and rotation-log changes with the declared
Jacobian columns. Run this independently for DART and the candidate backend. This catches a shared comparison bug caused
by applying the same incorrect row swap or frame conversion to both arrays.

**Validation 1D — dynamics invariants:** before CPU/GPU equality, assert that each mass matrix is finite, symmetric
within tolerance, and positive definite on valid unconstrained states. Check dimensions and articulated slicing of
`bias[6:]`, `(M * qdd)[6:]`, and effort limits. With injected `qdd = 0`, `motion_term` must equal the articulated bias
slice; with only one acceleration nonzero, independently multiply the matching mass-matrix column.

**Validation 1E — root-frame and batch equivariance:** evaluate each state alone and in a mixed batch, then permute the
batch. Apply seeded root translations and yaw rotations and verify the expected changes to world body poses, contact
poses, CoM, and world Jacobians while robot masses, effort limits, and joint ordering remain unchanged. Compare results
after mapping them into the same frame.

**Exit:** one selected backend supplies every required primitive in batches, and the primitive parity report passes the
finite-difference, dynamics-invariant, root-frame, and provisional numerical gates below for all supported robots.

### Stage 2 — Port contact geometry and LP assembly as pure tensor operations

Implement a batched, stateless version of the current rectangular contact model. Keep a single declared wrench layout
internally and convert at the API boundary if legacy non-SVA ordering must be supported.

#### Freeze the matrix schema

The first assembly test must validate the schema before validating any floating-point values. For the current
ZMP/CoM-velocity problem, define:

- `N`: number of active contacts in the compact CPU problem;
- `Q`: total DART degrees of freedom;
- `J = Q - 6`: articulated degrees of freedom;
- `R = 16`: contact-wrench-cone inequality rows per rectangular contact;
- `W = 6`: wrench variables per contact;
- `S = 4` when LIPM assumptions are enabled, otherwise `S = 0`.

The decision vector is `[wrench_0, ..., wrench_(N-1), output_x, output_y]`. Immediately after
`McZMPArea::constructProblem_`, the expected compact shapes are:

| Object | Shape | Meaning |
| --- | --- | --- |
| `F` | `[16N + 2J + 4, 6N + 2]` | Inequality left-hand side in `F x <= f` |
| `f` | `[16N + 2J + 4]` | Inequality right-hand side |
| `A` | `[S + 3, 6N + 2]` | Equality left-hand side in `A x = b` |
| `b` | `[S + 3]` | Equality right-hand side |

Record these logical row and column slices in the oracle manifest:

| Object | Logical block |
| --- | --- |
| `F[16i:16(i+1), 6i:6(i+1)]` | Inertial contact-wrench cone for active contact `i` |
| `F[16N:16N+J, 6i:6(i+1)]` | Upper torque block `-J_i^T` for articulated rows |
| `F[16N+J:16N+2J, 6i:6(i+1)]` | Lower torque block `+J_i^T` |
| Last four rows/last two columns of `F` | `+x`, `-x`, `+y`, `-y` projection-radius bounds |
| First `16N` entries of `f` | Zero contact-cone right-hand side |
| Next `J` entries of `f` | `tau_upper - motion_term` |
| Next `J` entries of `f` | `-tau_lower + motion_term` |
| Last four entries of `f` | Projection radius |
| First `S` rows of `A,b` | Optional LIPM assumptions |
| Next two rows of `A,b` | ZMP or CoM-velocity projection equation; output block is `-I` |
| Last row of `A,b` | Reserved line-constraint row, zero immediately after problem construction |

The CPU contact container is a `std::map`, while the GPU API uses fixed semantic contact slots. The oracle exporter must
store both the raw CPU order and the permutation into the canonical slot order. Never compare a compact CPU matrix with
a padded GPU matrix until the rows and columns have been remapped into the same logical layout.

Port and test, in this order:

1. local 16-by-6 wrench-cone construction;
2. rotation into the inertial frame;
3. resultant-wrench multiplier/grasp transformation;
4. torque-limit blocks from contact Jacobians;
5. LIPM equality blocks;
6. ZMP projection blocks;
7. CoM-velocity projection blocks;
8. active-contact masking/padding and search-radius bounds.

Use fixed-size padded tensors and masks for contact subsets. Do not compact contacts with a Python loop in the
simulation step. Assert finite values, positive mass/CoM height, valid friction, nonnegative contact sizes, and at least
one active contact before solving.

The matrix test corpus must cover the following axes. A pairwise design is acceptable for the larger random corpus, but
every listed corner case needs at least one named golden test.

| Axis | Required cases |
| --- | --- |
| Active contacts | Each individual slot; both feet; foot plus hand; all four; zero-contact invalid case |
| Contact convention | Non-SVA and SVA unit cases |
| LIPM assumptions | Enabled and disabled |
| Torque limits | All enabled, all disabled, and one disabled contact for `McZMPArea` |
| Friction | `0.0`, `0.3`, `0.7`, `1.0`, plus invalid-value rejection |
| Patch size | Square and asymmetric `halfX != halfY` to expose axis swaps |
| Contact transform | Identity, translation, 90-degree rotations, and a general seeded rotation |
| State | Reference, checked-in G1 cases, nonzero velocity, nonzero acceleration, and seeded valid poses |
| Projection | ZMP, controllable CoM velocity, stable CoM velocity, and two ZMP heights |
| Robot | Every repository robot that passes the Stage 0 load smoke test |

**Validation 2A — hand-computable contact cases:** test identity, pure translation, 90-degree rotation, zero friction,
and torque-limit-disabled cases. Verify signs using explicit feasible and infeasible wrenches, not only matrix equality.

**Validation 2B — shape, labels, and sparsity:** before elementwise comparison, assert the formulas above for every
compact CPU case and the declared fixed shapes for every padded GPU case. Compare:

- row and column labels after canonical remapping;
- active-contact and active-constraint masks;
- dtype, device, finite-value mask, and contiguity assumptions required by the solver;
- the exact structural-zero pattern, including off-diagonal contact-cone blocks;
- the final reserved equality row before any direction-specific mutation.

A schema, permutation, or sparsity failure is a distinct test failure and must not be reported as a large numerical
error.

**Validation 2C — primitive-to-block reconstruction:** calculate every block independently from the exported primitives
and compare it with the corresponding CPU matrix slice. At minimum, test:

- local CWC to inertial CWC rotation for each contact;
- resultant-wrench multiplier placement in the aggregate grasp matrix `G`;
- `-J_i^T` and `+J_i^T` articulated torque blocks;
- `motion_term = bias[6:] + (mass_matrix * acceleration)[6:]` and both torque-bound vectors;
- the four projection-radius rows and right-hand sides;
- the LIPM `B * G` block and its right-hand side;
- the height/CoM/mass/gravity projection factor;
- the angular-momentum cross term and the final `-I` output block.

This test should be able to report, for example, `F.torque_upper.leftFoot` rather than only `F differs`.

**Validation 2D — complete CPU/GPU matrix parity:** assemble from identical exported CPU primitives and compare the
canonicalized matrices element by element, before invoking either solver. For each of `F`, `f`, `A`, and `b`, report:

- maximum absolute and relative error with sample, row label, column label, CPU value, and GPU value;
- Frobenius/L2 error for context;
- count of entries outside tolerance;
- structural-zero mismatches and sign mismatches separately.

The minimum comparison is equivalent to:

```python
torch.testing.assert_close(F_gpu[active_F], F_cpu_padded[active_F], atol=atol, rtol=rtol)
torch.testing.assert_close(f_gpu[active_f], f_cpu_padded[active_f], atol=atol, rtol=rtol)
torch.testing.assert_close(A_gpu[active_A], A_cpu_padded[active_A], atol=atol, rtol=rtol)
torch.testing.assert_close(b_gpu[active_b], b_cpu_padded[active_b], atol=atol, rtol=rtol)
```

Do not allow NaNs to compare equal, and do not let large padded regions dilute error statistics.

**Validation 2E — row-semantic probes:** construct simple decision vectors with only one contact-wrench component or one
output component nonzero. Compare CPU and GPU values of `F x - f` and `A x - b` row by row. Then perturb one source
quantity at a time—friction, `halfX`, `halfY`, one Jacobian entry, one torque limit, CoM height, ZMP height, projection
radius, or angular-momentum term—and assert that only the expected blocks change.

**Validation 2F — cross-residual tests:** retain the complete CPU LP solution `x_cpu` for several cardinal and diagonal
directions. After GPU assembly, evaluate that same solution against both matrix sets. Once a GPU solution exists, do the
reverse evaluation as well:

```text
ineq_gpu_on_cpu = max(F_cpu * x_gpu - f_cpu)
eq_gpu_on_cpu   = max(abs(A_cpu * x_gpu - b_cpu))
ineq_cpu_on_gpu = max(F_gpu * x_cpu - f_gpu)
eq_cpu_on_gpu   = max(abs(A_gpu * x_cpu - b_gpu))
```

Both cross-residual directions must pass. This catches a GPU matrix/solver pair that is internally consistent but solves
a different problem from the CPU oracle.

**Validation 2G — ZMP/CoM-velocity algebraic invariants:** for the same state and contacts:

- `F_zmp == F_com_velocity` and `f_zmp == f_com_velocity` before padding;
- the LIPM rows of `A,b` are identical;
- for controllable CoM velocity, the wrench portion of the two projection rows and their right-hand side are the ZMP
  values scaled by `k = -omega / (pTwo - 1)`;
- the `-I` output block and reserved final row are not scaled;
- changing only `zmpHeight` changes the expected projection block but not `F` or `f`.

Test status 0, 1, and 2 directly even if production initially uses only status 1. These identities localize errors in
the projection mode without requiring a solver.

**Validation 2H — compact/padded equivalence:** solve the compact CPU matrix and its canonical padded representation
with the same CPU solver. Their statuses and 64 support values must agree. Fill inactive GPU slots with large finite
garbage values and verify that masks make the result unchanged. Test zero, one, and mixed active-contact counts in the
same batch, permute batch order, and permute physical storage slots while preserving semantic contact identifiers.

**Validation 2I — monotonic and metamorphic checks:** in addition to CPU equality, validate properties that both
backends should satisfy:

- increasing friction must not decrease any directional support beyond tolerance;
- disabling a torque-limit block in `McZMPArea` must not shrink the feasible projection;
- increasing an active projection-radius bound must not shrink the projection;
- swapping contact storage order and applying the corresponding row/column permutation must not change support values;
- with other exported primitives fixed, changing injected acceleration must affect the torque right-hand side through
  `M * qdd`, while changing the injected bias vector must affect only the corresponding torque right-hand side;
- asymmetric contact dimensions and a 90-degree contact rotation must expose the expected `halfX`/`halfY` axis swap.

Record exceptions where another active constraint makes a monotonic change exactly zero; zero change is valid, reversal
is not.

**Validation 2J — CPU-solver round trip:** copy GPU-assembled matrices back to the CPU in validation code only and solve
them with the same Stabiliplus/GLPK path as the oracle. Compare status, support value, feasibility residuals, and output
point for all 64 directions. This proves assembly parity before introducing a new solver. Host copies are allowed in
this offline test and remain forbidden in the Isaac Lab runtime.

Use the following differential ladder so each integration boundary has its own failure gate:

| Robot primitives | LP assembly | Solver | Purpose |
| --- | --- | --- | --- |
| CPU | CPU | CPU/GLPK | Authoritative golden result |
| Exported CPU | GPU tensor assembly | CPU/GLPK | Isolate GPU matrix assembly |
| Exported CPU matrices | None | GPU solver | Isolate GPU solver; Stage 3 |
| GPU backend | GPU tensor assembly | CPU/GLPK | Isolate robot primitives plus assembly |
| GPU backend | GPU tensor assembly | GPU solver | End-to-end result; Stage 4 |

Suggested independently runnable tests are `LPMatrixSchema_test`, `LPMatrixBlockParity_test`,
`LPMatrixCrossResidual_test`, `LPMatrixPaddingMask_test`, and `FrozenLPGpuSolver_test`. A failure in a later test should
print the most recent earlier gate that passed.

**Exit:** frozen CPU primitives fed into GPU tensor assembly reproduce all active entries of the oracle `F`, `f`, `A`,
and `b`; compact and padded forms are solver-equivalent; cross-residual and CPU-solver round-trip tests pass; and every
mismatch report names its logical block. This isolates constraint algebra from both robot kinematics and the new solver.

### Stage 3 — Validate the GPU projection solver on frozen CPU LPs

Select or implement a solver that supports the actual batched problem shape and fixed directional objectives.
Benchmark the solver with the repository's number of DOFs, 1–4 contacts, 64 directions, and production Isaac Lab batch
sizes before committing the full migration to it.

Start by sending the exported CPU `F`, `f`, `A`, and `b` directly to the candidate GPU solver. For each state and each
direction, record:

- solver status;
- objective/support value;
- primal equality residual;
- maximum inequality violation;
- iteration count, if available;
- whether the returned point is finite.

Do not compare only vertex arrays. Different solvers can return a different point along the same optimal edge. Compare
directional support values and feasibility first, then compare polygon geometry.

**Validation 3A — artifact ingestion:** load each frozen CPU matrix artifact and verify its checksum, schema version,
shape, row/column labels, contact permutation, active masks, dtype, and endianness. Solve it once with CPU/GLPK before
export and once after reload; statuses, support values, and residuals must agree. This detects serialization, transpose,
and row-major/column-major errors before exercising CUDA.

**Validation 3B — analytical LPs:** include boxes, triangles, redundant constraints, inactive padded constraints,
infeasible problems, unbounded problems, objectives parallel to an optimal edge, and problems with a single active
decision variable. Store known support values so this test does not depend on GLPK as its only oracle.

**Validation 3C — direction and objective convention:** solve `(+x, +y, -x, -y)` first, followed by the four diagonal
directions. Verify that the solver maximizes `direction dot output`, that the output variables are the final two active
columns, and that reversing a direction queries the opposite bound. Do not require equal magnitudes except in symmetric
analytical cases. Catch an objective-sign or output-column error here before running all 64 directions.

**Validation 3D — frozen CPU LP comparison:** solve every Stage 0 `F,f,A,b` set with both CPU/GLPK and the GPU solver in
float64. For every one of the 64 directions compare:

- feasible/infeasible/unbounded/numerical-failure status;
- support value `direction dot output`;
- `max(F x - f)` and `max(abs(A x - b))`, evaluated with both CPU and GPU matrices;
- returned output coordinates when the optimum is unique;
- primal/dual objective gap and KKT residual when the candidate solver exposes them.

Do not require the full decision vector to match when contact wrench distribution is non-unique. Require feasibility and
the projected support value instead.

**Validation 3E — row scaling, permutation, and redundancy:** positively scale individual inequality rows and their
right-hand sides, nonzero-scale equality rows and their right-hand sides, permute rows, and duplicate redundant rows.
After undoing metadata permutations, solver status and support values must remain unchanged. This validates that padding
and normalization do not accidentally alter the feasible set.

**Validation 3F — batch independence and failure containment:** combine well-conditioned, nearly degenerate,
infeasible, and unbounded frozen LPs in one batch. Compare each batched answer with the same LP solved alone, permute
the batch, and verify that a failed environment neither corrupts nor suppresses valid neighbors.

**Validation 3G — runtime dtype and conditioning:** after float64 passes, repeat the full corpus in float32. Report
error against both the float64 GPU result and CPU/GLPK, stratified by condition estimate, contact count, and active
constraint count. Exercise nearly parallel polygon edges and nearly active torque/friction limits. No failed solve may
be converted silently into a plausible polygon.

**Validation 3H — polygon invariants:** for every valid result, check finite vertices, counter-clockwise or declared
directional ordering, nonnegative area, convexity, constraint feasibility, and at least three unique support points.
Recompute all 64 supports from the returned polygon and compare them with the direct per-direction solver objectives.

**Exit:** the GPU solver passes all frozen-LP comparisons without using GPU kinematics or GPU LP assembly. Its failure
status is deterministic and is propagated to the output validity mask. Artifact ingestion, objective convention,
row-invariance, mixed-batch containment, and float32-conditioning reports must all pass independently.

### Stage 4 — Compose the exact batched GPU backend

Connect the validated robot primitives, contact/constraint assembly, and projection solver behind the proposed tensor
API. Keep a batch-one diagnostic program or script as the GPU equivalent of `AreaCalculator`; it should be a thin
client, not a second implementation.

**Validation 4A — end-to-end corpus:** compare CPU and GPU outputs over the golden corpus plus a larger seeded sample.
Use:

- validity/status agreement;
- all 64 directional support errors;
- symmetric Hausdorff distance;
- centroid distance;
- absolute and relative area error;
- inside/outside classification agreement for seeded query points/velocities away from the polygon boundary.

Ordered vertex equality is required only after both backends use the same fixed-direction postprocessor and duplicate
policy.

**Validation 4B — batch independence:** compare each element of a mixed batch with the result obtained when it is solved
alone. Permute the batch and contact-slot storage order and prove that outputs map back unchanged.

**Validation 4C — device/dtype:** run batch sizes 1, 2, 64, 1024, and the intended production environment count in
float64 validation mode and the chosen runtime dtype. Reject unsupported dtype/device combinations explicitly.

**Validation 4D — complete `AreaCalculator` parity, when required:** give `McComArea` a deterministic directional
oracle, then repeat Stages 2–4 for its different equality/projection equations. Add `com_position_vertices`, validity,
residual, and status outputs without coupling them to the first two areas. Do not infer this validation from ZMP parity.

**Exit:** end-to-end conformance passes on every supported robot and contact mode. The CPU backend remains available for
offline differential tests. If complete `AreaCalculator` equivalence is part of the selected scope, Validation 4D must
also pass.

### Stage 5 — Integrate with Isaac Lab

Add an adapter that maps Isaac Lab articulation/contact data to the exact tensor contract. Keep this adapter separate
from `LearnedStabilityReward` so experiments can select `cpu_oracle` (offline only), `gpu_exact`, or `gpu_learned`
without confusing their accuracy claims.

The hot path must:

- reuse preloaded robot/contact constants;
- accept and return tensors on `env.device`;
- avoid `.cpu()`, `.numpy()`, `.item()`, host synchronization, YAML/URDF parsing, and per-environment loops;
- expose validity and solver failures in metrics;
- give invalid environments a documented deterministic fallback reward;
- use the repository's joint/contact name mapping rather than relying on simulator order.

**Validation 5A — adapter mapping:** for one environment, snapshot the mapped root state, joint state, contacts,
friction, and body identifiers and compare them with a direct CPU-oracle input.

**Validation 5B — synchronized rollout:** replay a short deterministic Isaac Lab state trace through the offline CPU
oracle and the GPU backend. Compare every frame, including contact transitions and invalid states.

**Validation 5C — simulator smoke:** run at least one short rollout with all reward terms enabled. Assert finite
observations/rewards, stable output shapes, no increasing CUDA allocation, and no hidden host copies in a profiler
trace.

**Exit:** an Isaac Lab smoke run consumes exact GPU polygons entirely on device, and trace replay passes conformance.

### Stage 6 — Establish performance and memory gates

Measure before optimizing. Use warm-up iterations, CUDA events, explicit synchronization only around measurements, and
report median/p95 latency, throughput, peak allocated memory, solver failure rate, robot/DOF/contact counts, dtype, GPU,
CUDA/PyTorch versions, and direction count.

Benchmark these components separately:

1. robot primitives;
2. contact and LP assembly;
3. projection solve;
4. polygon validation/postprocessing;
5. complete Isaac Lab reward term.

Measure batch sizes 1, 64, 1024, 4096, and the production count. Compare with the CPU oracle at batch one and with a
CPU loop at a representative batch. The production acceptance target is:

- zero device-to-host transfers in the steady-state path;
- GPU throughput better than the CPU oracle at the production batch;
- the exact stability term consumes less than 10% of the measured Isaac Lab step time;
- memory use is bounded after warm-up and leaves enough headroom for the policy and simulator;
- numerical/failure gates remain satisfied under the benchmark workload.

If the 10% target is missed, profile before changing mathematics, precision, direction count, or tolerances. Any such
change creates a new accuracy/performance variant and requires a fresh conformance report.

**Exit:** a reproducible performance report meets the production gates on the target machine.

### Stage 7 — Dependency and packaging cleanup

Only after conformance and performance pass:

- isolate GPU code in its own optional target/package so CPU-only builds and tests still work;
- ensure the GPU runtime target does not link DART, Stabiliplus, GLPK, OSG, ImGui, or other CPU oracle/GUI libraries;
- keep one-time host parsing dependencies only if they do not enter the runtime dependency surface;
- pin tested CUDA, PyTorch, Isaac Lab, solver, and optional cuRobo versions;
- document the supported GPU architectures and a CPU-oracle developer setup;
- add CPU-only CI plus CUDA conformance/smoke CI where a runner is available.

Removing DART from the entire repository is not required. It remains useful for the oracle, fixtures, GUI, and
differential tests until a separate deprecation decision is made.

**Exit:** clean CPU-only configuration still passes, the optional GPU package installs reproducibly, and a dependency
inspection confirms that the Isaac Lab runtime does not load the CPU oracle stack.

### Optional Stage 8 — Visualization

If an interactive visualizer is still wanted, make `puppet` consume serialized or streamed GPU polygon results, or add a
small Isaac Lab debug visualization. Do not port DART IK/rendering merely to satisfy the word “GPU.” Validate displayed
vertices against the backend output and keep visualization out of performance measurements.

## Provisional numerical tolerances

These are starting gates, not permission to hide a systematic convention error. Tighten them after measuring the Stage 0
corpus. Use combined checks of `abs_error <= atol + rtol * abs(reference)`.

| Quantity | Float64 validation | Float32 runtime |
| --- | ---: | ---: |
| Matrix shape, labels, masks, and structural-zero pattern | Exact | Exact |
| Scalar/vector robot primitive | `atol=1e-9`, `rtol=1e-8` | `atol=2e-5`, `rtol=2e-5` |
| Transform translation / CoM | `1e-8 m` | `2e-5 m` |
| Transform rotation angle | `1e-8 rad` | `2e-5 rad` |
| Jacobian, mass, and bias entry | `atol=1e-8`, `rtol=1e-7` | `atol=5e-5`, `rtol=5e-5` |
| LP entry assembled from identical exported primitives | `atol=1e-10`, `rtol=1e-9` | `atol=2e-6`, `rtol=1e-5` |
| End-to-end LP entry including robot backend | `atol=1e-8`, `rtol=1e-7` | `atol=5e-5`, `rtol=5e-5` |
| Equality residual | `1e-8` | `1e-4` |
| Inequality violation | `1e-8` | `1e-4` |
| Directional support / Hausdorff distance | `1e-7` in output units | `2e-4` in output units |
| Centroid distance | `1e-7` in output units | `2e-4` in output units |
| Relative polygon area error | `1e-5` | `2e-3` |

ZMP and CoM-position coordinates use metres; CoM-velocity coordinates use metres per second. For inside/outside tests,
exclude queries within `5e-4` in the corresponding output units of the CPU boundary in float32 mode, then require at
least 99.9% classification agreement. Always report percentile and worst-case errors in addition to pass rates.

## Required deliverables

- Frozen mathematical/input/output contract and backend capability matrix.
- Deterministic CPU oracle exporter with labeled compact and canonical padded `F,f,A,b` artifacts.
- Compact golden corpus with checksums, active masks, contact permutations, CPU solutions, and residuals.
- Primitive, matrix-schema, matrix-block, cross-residual, padded-mask, frozen-LP, and end-to-end differential tests.
- Machine-readable matrix-parity report with worst sample/row/column attribution for every output area.
- Stateless batched GPU stability API with explicit validity/status diagnostics.
- Thin batch-one GPU `AreaCalculator` diagnostic client.
- Isaac Lab exact-backend adapter kept distinct from the learned TorchScript adapter.
- Numerical conformance report for every supported robot/contact mode.
- Reproducible latency/throughput/memory report on the target GPU.
- CPU-only and CUDA build/install/test documentation.

## Definition of done

The migration is complete when the exact GPU backend passes all staged validation gates, operates on Isaac Lab tensors
without steady-state host work, meets the measured performance budget, reports invalid/infeasible states safely, and can
be installed without the CPU oracle/GUI dependency stack. A GPU rewrite of the two executables alone does not satisfy
this definition.
