# Ranger Mini V3 model provenance

This directory contains a self-contained rolling-contact model of the AgileX Ranger Mini V3. It is an engineering
model for controller and simulator validation, not a manufacturer CAD model.

- The requested [SysWonder deployment repository](https://github.com/syswonder/robot-agilex-ranger_mini_v3) at
  revision `336067f195ab978456bf0bdc1b747e4f696931c2` identifies the chassis as four-wheel steer/drive and records a
  `0.74 m x 0.50 m` `base_link` footprint. Its `urdf/ranger_mini.urdf` identifies itself as a legacy Ranger Mini V2
  model, has no independent wheel-drive joints, and references `package://ranger_description/meshes/*.STL` files that
  the repository explicitly does not ship.
- AgileX's BSD-3-Clause [ranger_ros2](https://github.com/agilexrobotics/ranger_ros2) repository at revision
  `b6ea21a275ca5e7168130cc6470e61474681d679` defines the Ranger Mini V3 wheelbase as `0.494 m`, track as `0.364 m`,
  and the platform as four-wheel independent steering.
- AgileX's Ranger Mini 3.0 manual specifies a `0.72 m x 0.50 m x 0.345 m` envelope, 105 mm ground clearance, 75 kg
  curb mass, four 350 W hub drive motors, and four 100 W steering motors.

The model therefore uses the measured wheel-center layout (`x = +/-0.247 m`, `y = +/-0.182 m`), a 75 kg total
modeled mass, four steer joints, four independent drive joints, and a 0.125 m conventional-wheel radius consistent
with the published envelope. Primitive visual geometry deliberately replaces the unavailable mesh package. This
makes both the URDF and MuJoCo model visible and reproducible from this repository alone.

The Ranger floating-base/chassis frame is defined at the centre of the main shell. The link origin, inertial origin,
MuJoCo free-joint origin, and `FloatingBase` body sensor are all coincident at that point. The wheel axle plane is
35 mm below the frame; both descriptions apply this fixed offset to the wheel-bearing bodies and preserve the
original world-space wheel contacts at the 0.125 m initial height. Keeping this convention identical in URDF, MuJoCo,
and the `FloatingBase` body sensor prevents a visually displaced frame marker or a sensor pose that is offset from the
controller's `chassis` frame. MuJoCo includes a visual-only `chassis_frame` site at the same origin as a runtime aid.
