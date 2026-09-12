/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include <mc_observers/VelocityAidedTiltObserver.h>

#include <mc_observers/ObserverMacros.h>

#include <mc_control/MCController.h>

#include <mc_rtc/constants.h>
#include <mc_rtc/gui/Label.h>

#include <state-observation/tools/rigid-body-kinematics.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace mc_observers
{

void VelocityAidedTiltObserver::configure(const mc_control::MCController & ctl, const mc_rtc::Configuration & config)
{
  robot_ = config("robot", ctl.robot().name());
  if(!ctl.robots().hasRobot(robot_)) { mc_rtc::log::error_and_throw("[{}] No robot named {}", name(), robot_); }
  const auto & robot = ctl.robot(robot_);
  imuSensor_ = config("imuBodySensor", robot.bodySensor().name());
  if(!robot.hasBodySensor(imuSensor_))
  {
    mc_rtc::log::error_and_throw("[{}] No body sensor named {} in robot {}", name(), imuSensor_, robot_);
  }
  velocityFunction_ = config("velocityFunction", "VelocityAidedTilt::SensorVelocity::" + robot_);
  activationFunction_ = config("activationFunction", "VelocityAidedTilt::VelocityActivation::" + robot_);
  tiltKey_ = "VelocityAidedTilt::Tilt::" + robot_;
  tiltBodyKey_ = "VelocityAidedTilt::TiltBody::" + robot_;
  normalKey_ = "VelocityAidedTilt::Normal::" + robot_;

  // alpha / beta / gamma, and why these three numbers.
  //
  // THE ERROR MODEL. Write the filter of tilt-estimator.cpp around the truth,
  // with u = x1 - v the velocity-state error, d = x2' - x2_true the tilt error
  // and n the error in the velocity INPUT, yv = v + n. The true trajectory is an
  // exact particular solution - substituting (v, x2_true, x2_true) with the
  // strapdown accelerometer model ya = dv/dt + omega x v + g x2 makes every
  // residual vanish identically - so the error dynamics are homogeneous, and
  //
  //     d(s) / n(s) = -beta s / (s^2 + alpha s + beta g)
  //
  // with omega_n = sqrt(beta g) and zeta = alpha / (2 sqrt(beta g)). Three
  // consequences, all of which the sweep below confirms:
  //   - the DC gain is ZERO. A constant wheel-slip bias cannot bias the tilt;
  //     it is absorbed into x1. That is what makes a wheel-odometry input with
  //     an unknown slip offset admissible here at all.
  //   - the PEAK gain, at resonance, is exactly beta/alpha. This, not any lag,
  //     is what sets the error on a skidding vehicle.
  //   - there is NO steady lag behind a moving attitude. The plan this was
  //     built from predicted a ramp lag of alpha/(beta g) seconds; that came
  //     from treating ya as g * x2_true, which drops the dv/dt content of the
  //     accelerometer. With the full reading there is no such term, and the
  //     measurement below agrees: the fitted lag is 0.005-0.070 s against
  //     0.032-2.04 s predicted, and it moves the WRONG WAY with beta.
  //
  // THE SWEEP. This observer on the 10 degree ramp lane of ramp_terrain.xml,
  // loop still open (terrainNormal from configuration), forward at 0.3 m/s,
  // Tasks backend, torque regime, gamma = 10 throughout. "toe peak" is the
  // worst |estimated - measured| tilt within half a wheelbase of the concave
  // break at x = 0.6 m; "incline RMS" is over the constant-slope stretch above
  // it; "step" is the time this observer takes to reach 9.5 of a 10 degree
  // step commanded through the IMU alone (TiltEstimateIsNotAConstantTB).
  //
  //   alpha  beta   beta/alpha   toe peak   incline RMS   step to 9.5 deg
  //   ------------------------------------------------------------------
  //     5      1       0.20       0.465 deg   1.019 deg      1.185 s
  //     5      4       0.80       1.564       1.716          0.430 s
  //     5     16       3.20       4.060       3.416          0.225 s
  //    10      1       0.10       0.353       0.804          > 2 s (8.60 at 2 s)
  //    10      4       0.40       1.135       1.288          0.665 s
  //    10     16       1.60       3.268       2.368          0.270 s
  //    20      1       0.05       0.224       0.668          > 2 s (6.05 at 2 s)
  //    20      4       0.20       0.745       1.006          1.520 s
  //    20     16       0.80       2.369       1.692          0.430 s
  //
  // The error is monotone in beta/alpha and collapses on it: the two pairs that
  // share a ratio of 0.20 give 1.019 and 1.006, and the two that share 0.80 give
  // 1.716 and 1.692. It is the velocity-noise gain and nothing else.
  //
  // WHY 10 AND 4 AND NOT THE SWEEP'S BEST. That lane is the one case in the
  // whole ramp suite where the vehicle skids badly - 74% of its samples have a
  // wheel off the ground and its worst rolling slip is 2.5 m/s - so the sweep
  // measures only the noise limb of the trade-off. Tuning to it would be tuning
  // to a failing run. On the runs where the vehicle behaves, this observer at
  // (10, 4, 10) is already at 0.0006 deg RMS on a constant 10 degree slope and
  // 0.0012 deg on a 20 degree one, driving forward - two orders below anything
  // the gains could improve - and there it is bandwidth, not noise, that is at
  // stake. Among the three grid points with the textbook zeta = 0.80 -
  // (5, 1), (10, 4) and (20, 16), at omega_n = 3.13, 6.26 and 12.5 rad/s -
  // (10, 4) is the middle of the measured frontier: half the ramp error of
  // (20, 16) at a third of its noise gain, and two and a half times the step
  // response of (5, 1). beta = 1 is excluded outright at alpha >= 10: it puts
  // the slow pole at beta g / alpha <= 1 rad/s and the estimate is still 1.4 to
  // 3.9 degrees short of a 10 degree step two seconds after it is commanded.
  //
  // gamma only sets how fast the unit tilt x2 follows the intermediate x2'; at
  // 10 that is a 0.1 s pole, an order below the alpha/beta pair, and the sweep
  // above is insensitive to it.
  // Statement form: a non-const default selects Configuration's
  // write-into-reference overload, which returns void.
  config("alpha", alpha_);
  config("beta", beta_);
  config("gamma", gamma_);
  config("minimumActivation", minimumActivation_);
  config("updateRobot", updateRobot_);
  if(!std::isfinite(alpha_) || alpha_ <= 0.0 || !std::isfinite(beta_) || beta_ <= 0.0 || !std::isfinite(gamma_)
     || gamma_ <= 0.0 || !std::isfinite(minimumActivation_) || minimumActivation_ < 0.0)
  {
    mc_rtc::log::error_and_throw("[{}] alpha, beta and gamma must be finite and positive and minimumActivation "
                                 "finite and non-negative",
                                 name());
  }
  estimator_ = std::make_unique<stateObservation::TiltEstimator>(alpha_, beta_, gamma_, dt_);
  desc_ = name_ + " (imu=" + imuSensor_ + ", alpha=" + std::to_string(alpha_) + ", beta=" + std::to_string(beta_)
          + ", gamma=" + std::to_string(gamma_) + ")";
}

void VelocityAidedTiltObserver::reset(const mc_control::MCController & ctl)
{
  const auto & robot = ctl.robot(robot_);
  const auto & imu = robot.bodySensor(imuSensor_);
  // Seed the tilt from the pose the robot already has rather than from the
  // world vertical: a controller reset onto a slope would otherwise start the
  // filter a lane angle away from the truth and spend its whole settling time
  // recovering from an avoidable transient. posW().rotation() is the
  // world-to-body map E_0_b, so E_0_b * e_z - its third column - is the world
  // vertical in body coordinates, and X_b_s's rotation carries it to the
  // sensor frame.
  const Eigen::Matrix3d E_b_s = imu.X_b_s().rotation();
  tilt_ = E_b_s * ctl.realRobot(robot_).posW().rotation().col(2);
  if(!tilt_.allFinite() || tilt_.norm() < 1e-9) { tilt_ = Eigen::Vector3d::UnitZ(); }
  tilt_.normalize();
  tiltBody_ = E_b_s.transpose() * tilt_;
  tiltPrime_ = tilt_;
  imuVelocity_.setZero();
  sensorVelocity_.setZero();
  velocityValid_ = false;
  activation_ = 0.0;
  step_ = 0;
  estimator_->setAlpha(alpha_);
  estimator_->setBeta(beta_);
  estimator_->setGamma(gamma_);
  estimator_->setSamplingTime(dt_);
  estimator_->clearInputsAndMeasurements();
  estimator_->initEstimator(imuVelocity_, tiltPrime_, tilt_);
  run(ctl);
}

bool VelocityAidedTiltObserver::run(const mc_control::MCController & ctl)
{
  const auto & robot = ctl.robot(robot_);
  const auto & realRobot = ctl.realRobot(robot_);
  const auto & imu = robot.bodySensor(imuSensor_);
  // yg and ya are already in the sensor frame: an IMU measures its own angular
  // velocity and its own specific force, and mc_rtc stores both verbatim.
  const Eigen::Vector3d yg = imu.angularVelocity();
  const Eigen::Vector3d ya = imu.linearAcceleration();
  if(!yg.allFinite() || !ya.allFinite())
  {
    error_ = "body sensor " + imuSensor_ + " reported a non-finite gyro or accelerometer value";
    return false;
  }
  if(!ctl.datastore().has(velocityFunction_))
  {
    error_ = "no \"" + velocityFunction_
             + "\" function on the datastore; the controller must publish the measured body-frame velocity";
    return false;
  }

  // X_b_s maps the parent body frame to the sensor frame: E_b_s rotates a body
  // vector into the sensor, and r is the sensor origin in the body. The
  // velocity call reports the velocity of the BODY origin, so the lever arm has
  // to be added before the rotation. Both terms are identity/zero for the
  // Ranger's chassis IMU, and both are exercised by the unit test that gives it
  // a displaced, rotated sensor - the check that would have caught the JVRC1
  // mismatch between X_b_s and the site position.
  const Eigen::Matrix3d E_b_s = imu.X_b_s().rotation();
  const Eigen::Vector3d r_b_s = imu.X_b_s().translation();
  const Eigen::Vector3d omegaBody = E_b_s.transpose() * yg;
  const Eigen::Vector3d bodyVelocity =
      ctl.datastore().call<Eigen::Vector3d, const mc_rbdyn::Robot &>(velocityFunction_, realRobot);
  // NaN, not a sentinel number, when nothing reports the contact support: the
  // log then says "not reported" instead of inventing a value, and the gate
  // below falls through to trusting the velocity.
  const bool hasActivation = ctl.datastore().has(activationFunction_);
  activation_ = hasActivation ? ctl.datastore().call<double>(activationFunction_)
                              : std::numeric_limits<double>::quiet_NaN();
  velocityValid_ = bodyVelocity.allFinite() && (!hasActivation || activation_ >= minimumActivation_);
  if(velocityValid_) { sensorVelocity_ = E_b_s * (bodyVelocity + omegaBody.cross(r_b_s)); }
  else
  {
    // Feed the filter its own x1 back. The innovation yv - x1 is then exactly
    // zero, the tilt rows see nothing, and the estimate coasts on the gyro -
    // which is what an airborne or fully detached chassis actually warrants.
    sensorVelocity_ = imuVelocity_;
  }

  ++step_;
  estimator_->setMeasurement(sensorVelocity_, ya, yg, step_);
  const auto state = estimator_->getEstimatedState(step_);
  if(!state.allFinite())
  {
    error_ = "the tilt estimator diverged to a non-finite state";
    return false;
  }
  imuVelocity_ = state.segment<3>(0);
  tiltPrime_ = state.segment<3>(3);
  tilt_ = state.segment<3>(6);

  // Back to a world normal. x2 is the world vertical in the SENSOR frame;
  // E_b_s^T carries it to the parent body, where it is the "Rtez" of
  // state-observation's kinematics helpers - the local image of e_z under a
  // body-to-world rotation. posW().rotation() is the world-to-body map, so its
  // transpose is that body-to-world rotation and supplies the yaw this
  // estimator cannot.
  //
  // mergeTiltWithYawAxisAgnostic, not mergeRoll1Pitch1WithYaw2: the header at
  // rigid-body-kinematics.hpp:184 recommends it explicitly because the other
  // one throws at gimbal lock.
  tiltBody_ = E_b_s.transpose() * tilt_;
  const Eigen::Matrix3d bodyToWorld = realRobot.posW().rotation().transpose();
  normal_ = stateObservation::kine::mergeTiltWithYawAxisAgnostic(tiltBody_, bodyToWorld).col(2);
  tiltAngle_ = std::acos(std::clamp(normal_.z(), -1.0, 1.0));
  return true;
}

void VelocityAidedTiltObserver::publish(mc_control::MCController & ctl,
                                        const std::string & key,
                                        const Eigen::Vector3d & value)
{
  if(ctl.datastore().has(key)) { ctl.datastore().assign(key, value); }
  else { ctl.datastore().make<Eigen::Vector3d>(key, value); }
}

void VelocityAidedTiltObserver::update(mc_control::MCController & ctl)
{
  publish(ctl, tiltKey_, tilt_);
  publish(ctl, tiltBodyKey_, tiltBody_);
  publish(ctl, normalKey_, normal_);
  if(!updateRobot_) { return; }
  // Off by default and deliberately so: BodySensorObserver owns
  // realRobot().posW() in every configuration this ships in, and overwriting it
  // here would destroy the ground truth this estimator is validated against in
  // the same run. Only the attitude is written when it is on; the translation
  // is not this observer's to estimate.
  auto & realRobot = ctl.realRobot(robot_);
  const Eigen::Matrix3d bodyToWorld =
      stateObservation::kine::mergeTiltWithYawAxisAgnostic(tiltBody_, realRobot.posW().rotation().transpose());
  realRobot.posW(sva::PTransformd(bodyToWorld.transpose(), realRobot.posW().translation()));
}

void VelocityAidedTiltObserver::addToLogger(const mc_control::MCController &,
                                            mc_rtc::Logger & logger,
                                            const std::string & category)
{
  logger.addLogEntry(category + "_tilt", this, [this]() -> const Eigen::Vector3d & { return tilt_; });
  logger.addLogEntry(category + "_tiltBody", this, [this]() -> const Eigen::Vector3d & { return tiltBody_; });
  logger.addLogEntry(category + "_normal", this, [this]() -> const Eigen::Vector3d & { return normal_; });
  logger.addLogEntry(category + "_tiltAngle", this, [this]() { return tiltAngle_; });
  logger.addLogEntry(category + "_sensorVelocity", this,
                     [this]() -> const Eigen::Vector3d & { return sensorVelocity_; });
  logger.addLogEntry(category + "_imuVelocity", this, [this]() -> const Eigen::Vector3d & { return imuVelocity_; });
  logger.addLogEntry(category + "_velocityValid", this, [this]() { return velocityValid_; });
  logger.addLogEntry(category + "_velocityActivation", this, [this]() { return activation_; });
}

void VelocityAidedTiltObserver::addToGUI(const mc_control::MCController &,
                                         mc_rtc::gui::StateBuilder & gui,
                                         const std::vector<std::string> & category)
{
  gui.addElement(category, mc_rtc::gui::Label("Tilt angle [deg]", [this]()
                                              { return tiltAngle_ * 180.0 / mc_rtc::constants::PI; }),
                 mc_rtc::gui::Label("Velocity valid", [this]() { return velocityValid_; }));
}

} // namespace mc_observers

EXPORT_OBSERVER_MODULE("VelocityAidedTilt", mc_observers::VelocityAidedTiltObserver)
