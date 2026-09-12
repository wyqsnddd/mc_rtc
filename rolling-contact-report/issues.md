## Resolved rolling-contact issues

### 1. Robot selection used the wrong arity

The rolling-contact examples previously passed the implementation, description path, and variant as a three-item
`MainRobot` sequence. This did not match the one-argument robot alias convention used by `/usr/local/etc/mc_rtc.yaml`.
All checked-in examples now use one argument:

```yaml
MainRobot: RollingContactDifferential
MainRobot: RollingContactRangerMiniV3
```

The aliases resolve to the tracked `RollingContact` module and its differential or Ranger Mini V3 variant. The
robot-loader tests assert the one-argument parameter list and canonical variant for every rolling robot.

### 2. mc_mujoco did not show the four-steer robot

The requested Ranger Mini V3 reference was validated against
<https://github.com/syswonder/robot-agilex-ranger_mini_v3>. Its checked-in URDF is a legacy V2 description that points
to meshes not shipped by that repository, so the mc_rtc model is intentionally self-contained: primitive visible
geometries, four steering joints, four drive joints, and tracked wheel dimensions. The MuJoCo model explicitly
excludes chassis/wheel self-collision (the wheels sit inside the body envelope), preserving the four terrain contacts
needed by the rolling QP. Static MuJoCo/RBDyn parity and Tasks/TVM forward, crab, Ackermann, and pure-yaw runs pass.

### 3. Interactive control was missing

`mc_rtc-ranger-mini-v3-keyboard.yaml` now enables the optional RoboticsUtils `KeyboardCapture` adapter in the
`RollingContact` controller. From a real terminal, use W/S for forward/reverse, A/D for left/right crab motion,
Q/E for left/right yaw, `space` to clear all axes, and `x` to stop capture. Because terminal input is event-based, the
controller latches commands atomically; W/S/A/D clear a previous Q/E yaw while Q/E can be added to an active
translation for mixed turning. On a Q/E-to-W/S/A/D handoff, all four steering hinges are aligned before any drive
rate is released, preventing an asymmetric impulse while half the chassis still follows the old steering direction.
It feeds the resulting wheel rates to the posture task on the next control cycle. It
exposes keyboard status and reference-speed log entries for verification; close the MuJoCo window or press
Ctrl-C to terminate the simulator.
