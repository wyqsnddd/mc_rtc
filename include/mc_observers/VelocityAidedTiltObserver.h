/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include <mc_observers/Observer.h>
#include <mc_observers/api.h>

#include <state-observation/observer/tilt-estimator.hpp>

#include <Eigen/Core>

#include <memory>
#include <string>

namespace mc_observers
{

/** Roll and pitch of a body-mounted IMU, aided by an externally measured linear velocity.
 *
 * Wraps `stateObservation::TiltEstimator` (Benallegue et al., "Lyapunov-stable
 * orientation estimator for humanoid robots", RA-L 2020), whose state is
 *
 *   x1   the linear velocity of the IMU, expressed in the IMU frame
 *   x2'  a fast intermediate estimate of the tilt
 *   x2   the unit tilt: THE WORLD VERTICAL EXPRESSED IN THE IMU FRAME
 *
 * **Yaw is not estimated, and this observer does not claim to estimate it.**
 * A gyroscope and an accelerometer carry no information about heading: the
 * filter is driven by the direction of gravity, which is invariant under
 * rotation about itself. `Normal` below merges the estimated tilt with the yaw
 * of the robot it is given, and that yaw comes from wherever that robot's pose
 * comes from - here, the `FloatingBase` body sensor. Only `Tilt` is this
 * observer's own output.
 *
 * ### Why a velocity input, and why it must not come from the QP
 *
 * A driving robot's accelerometer measures specific force, which on a turn is
 * dominated by the transport term omega x v and not by gravity. `setMeasurement`
 * takes a velocity alongside the IMU precisely to cancel it: at convergence the
 * filter satisfies `g x2' = ya + yv x yg`. That removes the need for any
 * hand-rolled low-pass on the accelerometer - the alpha/beta structure IS the
 * complementary filter.
 *
 * The velocity is read through a datastore call so that the estimate stays
 * independent of the controller's own solution. Feeding it the QP's predicted
 * chassis velocity would make the filter confirm the model it is supposed to be
 * measured against.
 *
 * ### Datastore contract
 *
 * | dir | key                                            | type                                     |
 * |-----|------------------------------------------------|------------------------------------------|
 * | in  | `VelocityAidedTilt::SensorVelocity::<robot>`   | `Eigen::Vector3d(const mc_rbdyn::Robot &)` |
 * | in  | `VelocityAidedTilt::VelocityActivation::<robot>` | `double()`, optional                   |
 * | out | `VelocityAidedTilt::Tilt::<robot>`             | `Eigen::Vector3d` - x2, sensor frame     |
 * | out | `VelocityAidedTilt::TiltBody::<robot>`         | `Eigen::Vector3d` - x2, parent-body frame |
 * | out | `VelocityAidedTilt::Normal::<robot>`           | `Eigen::Vector3d` - body +z in the world |
 *
 * `TiltBody` is `Tilt` carried over by `X_b_s`, and it is the one a consumer
 * usually wants: it is the third column of the IMU parent body's `E_0_b`, so
 * `mc_rbdyn::rpyFromMat`'s roll and pitch are read straight off it, and the
 * same two expressions applied to a reference pose give a like-for-like
 * comparison. `Normal` cannot substitute for it - that is `E_0_b`'s third ROW,
 * a different vector.
 *
 * The input is the planar chassis velocity in the FLOATING-BASE BODY frame,
 * `[vx, vy, 0]`; this observer maps it to the sensor frame itself, lever arm
 * and rotation included, from the body sensor's own `X_b_s`.
 *
 * The optional activation call reports how much contact support that velocity
 * rests on, in wheels. Below `minimumActivation` the velocity is replaced by the
 * filter's own `x1`, which makes the innovation identically zero and degrades
 * the filter to gyro dead reckoning - the correct behaviour for an airborne
 * robot, and the reason a zero twist from a detached chassis is not read as
 * "standing still". A non-finite velocity is treated the same way. When the call
 * is absent the velocity is always trusted.
 *
 * The outputs are only published by `update()`, which the pipeline calls solely
 * for an entry declared `update: true`; `run()` receives a const controller and
 * cannot write the datastore.
 *
 * ### This observer does not own any robot's pose
 *
 * `updateRobot` defaults to **false**. `BodySensorObserver` owns
 * `realRobot().posW()` in every configuration this ships in, and the validation
 * of this estimator needs that pose intact, as ground truth, in the same run.
 */
struct MC_OBSERVER_DLLAPI VelocityAidedTiltObserver : public Observer
{
  VelocityAidedTiltObserver(const std::string & type, double dt) : Observer(type, dt) {}

  void configure(const mc_control::MCController & ctl, const mc_rtc::Configuration & config) override;
  void reset(const mc_control::MCController & ctl) override;
  bool run(const mc_control::MCController & ctl) override;
  void update(mc_control::MCController & ctl) override;

  /** x2: the world vertical expressed in the IMU frame. Unit, yaw-free. */
  const Eigen::Vector3d & tilt() const noexcept { return tilt_; }
  /** x2 carried over to the IMU's parent body frame. Unit, yaw-free. */
  const Eigen::Vector3d & tiltBody() const noexcept { return tiltBody_; }
  /** The estimated body +z axis of the IMU's parent body, in world coordinates. */
  const Eigen::Vector3d & normal() const noexcept { return normal_; }
  /** The velocity measurement actually fed to the filter, in the sensor frame. */
  const Eigen::Vector3d & sensorVelocity() const noexcept { return sensorVelocity_; }
  /** False while the velocity input was unusable and the filter dead reckoned. */
  bool velocityValid() const noexcept { return velocityValid_; }
  /** Angle between the estimated normal and the world vertical, in radians. */
  double tiltAngle() const noexcept { return tiltAngle_; }

protected:
  void addToLogger(const mc_control::MCController & ctl, mc_rtc::Logger & logger, const std::string & category) override;
  void addToGUI(const mc_control::MCController & ctl,
                mc_rtc::gui::StateBuilder & gui,
                const std::vector<std::string> & category) override;

  /** Publish @p value under @p key, creating the entry on the first cycle. */
  static void publish(mc_control::MCController & ctl, const std::string & key, const Eigen::Vector3d & value);

  std::string robot_; ///< Robot carrying the IMU (default: main robot)
  std::string imuSensor_; ///< BodySensor holding the gyro and accelerometer readings
  std::string velocityFunction_; ///< Datastore call supplying the body-frame velocity
  std::string activationFunction_; ///< Optional datastore call supplying the contact support
  std::string tiltKey_; ///< Datastore key this observer publishes x2 under
  std::string tiltBodyKey_; ///< Datastore key this observer publishes x2's body image under
  std::string normalKey_; ///< Datastore key this observer publishes the world normal under

  double alpha_ = 10.0; ///< Convergence gain of the IMU-frame linear velocity
  double beta_ = 4.0; ///< Fast convergence gain of the tilt
  double gamma_ = 10.0; ///< Orthogonality gain of the unit tilt
  double minimumActivation_ = 1.0; ///< Contact support, in wheels, below which the velocity is dropped
  bool updateRobot_ = false; ///< Write the estimated attitude into realRobot().posW()

  std::unique_ptr<stateObservation::TiltEstimator> estimator_;
  stateObservation::TimeIndex step_ = 0;

  Eigen::Vector3d tilt_ = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d tiltBody_ = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d tiltPrime_ = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d normal_ = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d imuVelocity_ = Eigen::Vector3d::Zero(); ///< x1, the filter's own IMU-frame velocity
  Eigen::Vector3d sensorVelocity_ = Eigen::Vector3d::Zero(); ///< yv, as fed to the filter
  double tiltAngle_ = 0.0;
  double activation_ = 0.0;
  bool velocityValid_ = false;
};

} // namespace mc_observers
