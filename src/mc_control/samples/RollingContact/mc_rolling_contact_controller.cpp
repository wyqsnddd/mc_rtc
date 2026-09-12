/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include "mc_rolling_contact_controller.h"

#include <mc_rbdyn/CylindricalSurface.h>
#include <mc_rtc/constants.h>
#include <mc_rtc/gui/ArrayLabel.h>
#include <mc_rtc/gui/Arrow.h>
#include <mc_rtc/gui/Button.h>
#include <mc_rtc/gui/Label.h>
#include <mc_rtc/gui/NumberInput.h>
#include <mc_rtc/gui/plot.h>
#include <mc_rtc/logging.h>
#include <mc_solver/TasksQPSolver.h>

#include <mc_tvm/Robot.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <atomic>
#include <numeric>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <limits>
#include <map>
#include <sstream>

#ifdef MC_ROLLING_CONTACT_HAS_KEYBOARD
#  include <RoboticsUtils/KeyboardCapture.h>
#endif

namespace mc_control
{

namespace
{

mc_solver::RollingContactLongitudinal longitudinalMode(const mc_rtc::Configuration & config)
{
  const std::string value = config("longitudinal", std::string{"hard"});
  if(value == "hard") { return mc_solver::RollingContactLongitudinal::Hard; }
  if(value == "soft") { return mc_solver::RollingContactLongitudinal::Soft; }
  mc_rtc::log::error_and_throw<std::invalid_argument>("RollingContact longitudinal must be hard or soft, got {}",
                                                      value);
}

/** twistWeight may be an explicit [w_vx, w_vy, w_omega] or a single scalar,
 * broadcast to all three axes. The scalar form is accepted for convenience
 * but deprecated: it reintroduces exactly the unit mismatch
 * eq:planar-twist-weight exists to remove, since one number cannot
 * simultaneously be correct in (m/s)^-2 and (rad/s)^-2 units.
 */
Eigen::Vector3d readTwistWeight(const mc_rtc::Configuration & config)
{
  if(!config.has("twistWeight")) { return Eigen::Vector3d::Ones(); }
  if(config("twistWeight").isNumeric())
  {
    const double scalar = config("twistWeight", 1.0);
    mc_rtc::log::warning(
        "RollingContact twistWeight given as a scalar ({}); this applies the same weight to vx, vy and omega "
        "even though they carry different units - (m/s)^-2 for vx/vy, (rad/s)^-2 for omega - and is deprecated. "
        "Prefer the explicit [w_vx, w_vy, w_omega] form.",
        scalar);
    return Eigen::Vector3d::Constant(scalar);
  }
  return config("twistWeight", Eigen::Vector3d{1.0, 1.0, 1.0});
}

} // namespace

struct MCRollingContactController::KeyboardInput
{
  bool start(double linearVelocity, double angularVelocity, int pollIntervalMs, char exitKey)
  {
#ifdef MC_ROLLING_CONTACT_HAS_KEYBOARD
    const char normalizedExitKey = static_cast<char>(std::tolower(static_cast<unsigned char>(exitKey)));
    // KeyboardCapture reports key events, but its internal command is derived
    // from only the key read during the current poll. A normal terminal sends
    // one byte per press (and does not report releases), so preserve each
    // command here until another command or the operator stops the command.
    capture.setKeyPressCallback([this, linearVelocity, angularVelocity, normalizedExitKey](char key)
                                 { handleKey(key, linearVelocity, angularVelocity, normalizedExitKey); });
    // Handle the exit key in the controller thread. RoboticsUtils invokes the
    // key callback from its polling thread, so enabling its built-in exit
    // path here would make that thread attempt to join itself during stop().
    RoboticsUtils::KeyboardCapture::Config config(
        linearVelocity, angularVelocity, pollIntervalMs, false, normalizedExitKey);
    return capture.start(config);
#else
    static_cast<void>(linearVelocity);
    static_cast<void>(angularVelocity);
    static_cast<void>(pollIntervalMs);
    static_cast<void>(exitKey);
    return false;
#endif
  }

  std::array<double, 3> command()
  {
#ifdef MC_ROLLING_CONTACT_HAS_KEYBOARD
    // The key callback runs on KeyboardCapture's worker. Defer stop() to this
    // controller-thread poll so the worker can exit and be joined safely.
    if(stopRequested.exchange(false)) { capture.stop(); }
    // Treat a stopped capture as an explicit zero command so the robot cannot
    // continue driving after the controller-side exit key has been processed.
    if(!capture.isRunning()) { return {0.0, 0.0, 0.0}; }
    return {vx.load(), vy.load(), wz.load()};
#else
    return {0.0, 0.0, 0.0};
#endif
  }

  bool running() const
  {
#ifdef MC_ROLLING_CONTACT_HAS_KEYBOARD
    return capture.isRunning();
#else
    return false;
#endif
  }

  void clear()
  {
#ifdef MC_ROLLING_CONTACT_HAS_KEYBOARD
    vx.store(0.0);
    vy.store(0.0);
    wz.store(0.0);
#endif
  }

  std::string status() const
  {
#ifdef MC_ROLLING_CONTACT_HAS_KEYBOARD
    // KeyboardCapture's status describes only the byte seen during its latest
    // poll. The controller deliberately latches commands until another
    // command, space, or x arrives, so expose that authoritative latched state
    // to the GUI/log instead of the transient helper state.
    const double forward = vx.load();
    const double lateral = vy.load();
    const double yaw = wz.load();
    std::ostringstream status;
    status << "W:" << (forward > 0.0 ? "1" : "0") << " S:" << (forward < 0.0 ? "1" : "0")
           << " A:" << (lateral < 0.0 ? "1" : "0") << " D:" << (lateral > 0.0 ? "1" : "0")
           << " Q:" << (yaw < 0.0 ? "1" : "0") << " E:" << (yaw > 0.0 ? "1" : "0");
    return status.str();
#else
    return "unavailable";
#endif
  }

#ifdef MC_ROLLING_CONTACT_HAS_KEYBOARD
  void handleKey(char key, double linearVelocity, double angularVelocity, char exitKey)
  {
    key = static_cast<char>(std::tolower(static_cast<unsigned char>(key)));
    if(key == exitKey || key == ' ')
    {
      vx.store(0.0);
      vy.store(0.0);
      wz.store(0.0);
      if(key == exitKey) { stopRequested.store(true); }
      return;
    }
    // Each translation key names one axis of the chassis twist and is an
    // explicit, complete command on the two others: W/S mean "drive forward or
    // back", not "drive forward and keep whatever else was running".
    //
    // Terminal input has no key-release event, so every axis this handler does
    // not rewrite stays latched and is indistinguishable from an axis the
    // operator is still holding down. Rewriting only the key's own axis makes
    // the latched remainder silently steer the robot: a previously latched Q/E
    // turns a requested straight or crab motion into an unintended circle, and
    // a previously latched A/D turns a requested straight motion into a
    // permanent diagonal. The second one is not a small effect - both
    // translation axes are commanded at the same keyboardLinearSpeed, so the
    // latched pair is always (v, v) and the robot travels at exactly 45 degrees
    // to the direction it points, for as long as the session lasts, with no key
    // other than space or the exit key able to clear it.
    //
    // Q/E deliberately keep the translation in effect: "add yaw to translation"
    // is what makes W then E an arc and A then Q a crabbing turn, and it is a
    // command the operator can undo with any translation key.
    switch(key)
    {
      case 'w':
        vx.store(linearVelocity);
        vy.store(0.0);
        wz.store(0.0);
        break;
      case 's':
        vx.store(-linearVelocity);
        vy.store(0.0);
        wz.store(0.0);
        break;
      case 'a':
        vx.store(0.0);
        vy.store(-linearVelocity);
        wz.store(0.0);
        break;
      case 'd':
        vx.store(0.0);
        vy.store(linearVelocity);
        wz.store(0.0);
        break;
      case 'q':
        wz.store(-angularVelocity);
        break;
      case 'e':
        wz.store(angularVelocity);
        break;
      default:
        break;
    }
  }

  RoboticsUtils::KeyboardCapture capture;
  std::atomic<double> vx{0.0};
  std::atomic<double> vy{0.0};
  std::atomic<double> wz{0.0};
  std::atomic<bool> stopRequested{false};
#endif
};

size_t MCRollingContactController::wheelIndex(const std::string & name, const char * query) const
{
  const auto wheel = std::find_if(wheels_.begin(), wheels_.end(),
                                  [&name](const auto & candidate) { return candidate.name == name; });
  if(wheel == wheels_.end())
  {
    throw std::invalid_argument(std::string{"Unknown rolling-contact "} + query + " wheel: " + name);
  }
  return static_cast<size_t>(std::distance(wheels_.begin(), wheel));
}

void MCRollingContactController::makeWheelValueCall(const std::string & key,
                                                    const char * query,
                                                    const std::vector<double> & values)
{
  datastore().make_call(key, [this, query, &values](const std::string & name) -> double
                        { return values[wheelIndex(name, query)]; });
}

MCRollingContactController::MCRollingContactController(mc_rbdyn::RobotModulePtr robotModule,
                                                       double dt,
                                                       const mc_rtc::Configuration & config)
: MCRollingContactController(robotModule, dt, config, Backend::Tasks)
{
}

MCRollingContactController::MCRollingContactController(mc_rbdyn::RobotModulePtr robotModule,
                                                       double dt,
                                                       const mc_rtc::Configuration & config,
                                                       Backend backend)
: MCController(robotModule, dt, config, backend)
{
  if(robot().name() != "rolling_diff" && robot().name() != "rolling_4s" && robot().name() != "ranger_mini_v3")
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>(
        "RollingContact controller requires rolling_diff, rolling_4s, or ranger_mini_v3, got {}", robot().name());
  }
  fourSteering_ = robot().name() != "rolling_diff";
  const auto settings = config.has("RollingContact") ? config("RollingContact") : mc_rtc::Configuration{};
  scenario_ = settings("scenario", std::string{"hold"});
  closedLoopFeedback_ = settings("closedLoopFeedback", false);
  positionFeedbackGain_ = settings("positionFeedbackGain", 5.0);
  maxYawTargetError_ = settings("maxYawTargetError", 0.5 * 3.14159265358979323846);
  // Strictly below pi, and with room to spare: sva::rotationError() switches to
  // its near-pi branch at |error| = pi - 1.105e-2 rad and returns NaN there,
  // which fails the QP outright. Reject a value that would let the reference
  // reach that configuration rather than discovering it at run time.
  if(!std::isfinite(maxYawTargetError_) || maxYawTargetError_ <= 0.0
     || maxYawTargetError_ > 0.9 * 3.14159265358979323846)
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>(
        "RollingContact maxYawTargetError must be finite, positive and at most 0.9 * pi; sva::rotationError is "
        "singular at pi and a heading target allowed to reach it makes the QP fail");
  }
  linearSpeed_ = settings("linearSpeed", 0.2);
  yawRate_ = settings("yawRate", 0.35);
  steeringAngle_ = settings("steeringAngle", 0.3);
  commandPeriod_ = settings("commandPeriod", 4.0);
  recoverySpeed_ = settings("recoverySpeed", 0.05);
  recoveryResidual_ = settings("recoveryResidual", 0.2);
  recoveryTransverseResidual_ = settings("recoveryTransverseResidual", 0.5);
  rollingWeight_ = settings("rollingWeight", 1000.0);
  recoveryRollingWeight_ = settings("recoveryRollingWeight", 1e6);
  // Ceiling on how fast the rolling-rate reference itself may change, so a step
  // in the commanded twist cannot ask the QP for the whole rate on the very
  // next cycle. It must stay well above the steering slope: the reference is a
  // projection onto the *measured* hinge heading, so a drive reference that
  // changes more slowly than the hinge swings keeps the wheels rolling in a
  // stale direction. This bounds every four-steering scenario, not just the
  // keyboard one it was originally named for, hence the rename; the
  // keyboard-scoped spelling remains a deprecated alias. It has no effect on a
  // differential chassis, whose branch of updateReference() sets the drive
  // targets straight from the commanded twist with no rate limiting.
  //
  // This was briefly widened to 100 rad/s^2, tuned only against the
  // kinematic ticker (no contact physics), where it halves the lateral slip
  // across a commanded-twist step (0.291 -> 0.079 m/s). Against closed-loop
  // MuJoCo contact physics that value saturates drive torque at the 35 Nm
  // limit and drives the min normal force to 0 on four-crab and
  // four-ackermann-left. A sweep against MuJoCo (four-crab, Tasks backend)
  // shows a sharp knee, not a gradual tradeoff: max drive torque is ~0.06 Nm
  // and min normal force ~184 N for driveAcceleration in [20, 40], then at 50
  // it jumps to 23 Nm / 0 N with the wheels 90% of the run in contact
  // fallback, and by 80-100 it is pinned at the 35 Nm limit with the wheels
  // detached 95%+ of the run. 20 rad/s^2 keeps a 2x margin below that knee and
  // is the value validated pre-rewrite (commit dff9cc26f1), where all four
  // MuJoCo cases and the ackermann_left/ackermann_right mirror symmetry pass.
  driveAcceleration_ = settings("driveAcceleration", 20.0);
  if(settings.has("keyboardDriveAcceleration"))
  {
    mc_rtc::log::warning("RollingContact keyboardDriveAcceleration is deprecated and now bounds every scenario's "
                         "drive reference; rename it to driveAcceleration");
    if(!settings.has("driveAcceleration")) { driveAcceleration_ = settings("keyboardDriveAcceleration", 20.0); }
  }
  if(!std::isfinite(driveAcceleration_) || driveAcceleration_ <= 0.0)
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>(
        "RollingContact driveAcceleration must be finite and positive");
  }
  if(!std::isfinite(commandPeriod_) || commandPeriod_ <= 0.0 || !std::isfinite(positionFeedbackGain_)
     || positionFeedbackGain_ < 0.0 || !std::isfinite(linearSpeed_)
     || !std::isfinite(yawRate_) || !std::isfinite(steeringAngle_) || !std::isfinite(recoverySpeed_)
     || recoverySpeed_ < 0.0 || !std::isfinite(recoveryResidual_) || recoveryResidual_ < 0.0
     || !std::isfinite(recoveryTransverseResidual_) || recoveryTransverseResidual_ < 0.0
     || !std::isfinite(rollingWeight_) || rollingWeight_ <= 0.0 || !std::isfinite(recoveryRollingWeight_)
     || recoveryRollingWeight_ < rollingWeight_)
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>(
        "RollingContact commandPeriod and rolling weights must be positive, positionFeedbackGain and recovery "
        "thresholds non-negative, "
        "recoveryRollingWeight must be at least rollingWeight, and command scalars must be finite");
  }
  const std::vector<std::string> differentialScenarios = {"hold",       "reverse",      "forward",
                                                           "turn_left",  "turn_right",   "circle_left",
                                                           "circle_right", "sinusoid",   "unequal_radii",
                                                           "infeasible_soft", "mode_cycle"};
  const std::vector<std::string> steeringScenarios = {"hold",       "forward",       "reverse",
                                                       "crab",       "ackermann_left", "ackermann_right",
                                                       "pure_yaw",   "steering_rate", "incompatible",
                                                       "mode_cycle", "keyboard"};
  const auto & supported = fourSteering_ ? steeringScenarios : differentialScenarios;
  if(std::find(supported.begin(), supported.end(), scenario_) == supported.end())
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>("Unknown RollingContact scenario '{}' for {}", scenario_,
                                                        robot().name());
  }

  wheels_ = makeWheels();
  wheelOffsets_.reserve(wheels_.size());
  const auto & chassis = robot().frame("chassis").position();
  for(const auto & wheel : wheels_)
  {
    const Eigen::Vector3d worldOffset =
        robot().frame(wheel.carrierFrame).position().translation() - chassis.translation();
    wheelOffsets_.push_back((chassis.rotation() * worldOffset).head<2>());
  }
  if(fourSteering_)
  {
    // Take the hinge rate ceiling from the model rather than restating the
    // URDF here, so a model change cannot silently desynchronize it.
    const double modelSteeringRate = [this]()
    {
      double rate = std::numeric_limits<double>::infinity();
      for(const auto & wheel : wheels_)
      {
        const auto joint = robot().jointIndexByName(wheel.steeringJoint);
        rate = std::min({rate, std::abs(robot().vl()[joint][0]), std::abs(robot().vu()[joint][0])});
      }
      return rate;
    }();
    // A non-const default would select Configuration's write-into-reference
    // overload, which returns void.
    maxSteeringRate_ = settings("maxSteeringRate", modelSteeringRate);
    // Convergence time of a steering hinge towards its reference heading. The
    // largest possible re-heading is pi/2, so this value sets the initial
    // demand for that worst case at pi/2 / 0.15 = 10.5 rad/s, just above the
    // 8 rad/s hinge ceiling: the fastest first-order law that still leaves the
    // hinge saturated only for the first instants of the widest swing. Slowing
    // it down is measurably worse - at half this slope the lateral slip across
    // a commanded-twist step doubled, from 0.085 to 0.19 m/s.
    steeringTimeConstant_ = settings("steeringTimeConstant", 0.15);
    if(!std::isfinite(maxSteeringRate_) || maxSteeringRate_ <= 0.0 || !std::isfinite(steeringTimeConstant_)
       || steeringTimeConstant_ <= 0.0)
    {
      mc_rtc::log::error_and_throw<std::invalid_argument>(
          "RollingContact maxSteeringRate and steeringTimeConstant must be finite and positive; the steering "
          "joint velocity limits supply the default rate");
    }
  }
  if(scenario_ == "keyboard")
  {
    const double keyboardLinearSpeed = settings("keyboardLinearSpeed", 0.3);
    const double keyboardAngularSpeed = settings("keyboardAngularSpeed", 0.5);
    const int keyboardPollIntervalMs = settings("keyboardPollIntervalMs", 10);
    const std::string keyboardExitKey = settings("keyboardExitKey", std::string{"x"});
    keyboardYawFeedbackGain_ = settings("keyboardYawFeedbackGain", 0.5);
    if(!std::isfinite(keyboardLinearSpeed) || keyboardLinearSpeed <= 0.0 || !std::isfinite(keyboardAngularSpeed)
       || keyboardAngularSpeed <= 0.0 || keyboardPollIntervalMs <= 0 || keyboardExitKey.size() != 1
       || !std::isfinite(keyboardYawFeedbackGain_) || keyboardYawFeedbackGain_ < 0.0)
    {
      mc_rtc::log::error_and_throw<std::invalid_argument>(
          "RollingContact keyboard speeds and poll interval must be positive, keyboardExitKey one character, "
          "and keyboardYawFeedbackGain non-negative");
    }
    keyboard_ = std::make_unique<KeyboardInput>();
    if(!keyboard_->start(keyboardLinearSpeed, keyboardAngularSpeed, keyboardPollIntervalMs, keyboardExitKey[0]))
    {
#ifdef MC_ROLLING_CONTACT_HAS_KEYBOARD
      mc_rtc::log::error_and_throw<std::runtime_error>(
          "RollingContact keyboard capture requires an interactive terminal on standard input");
#else
      mc_rtc::log::error_and_throw<std::runtime_error>(
          "RollingContact was built without RoboticsUtils::Keyboard; install RoboticsUtils and rebuild mc_rtc");
#endif
    }
    mc_rtc::log::info("RollingContact keyboard: W/S/A/D each command one translation axis and clear the other two "
                      "(so W after A drives straight forward, not diagonally); Q/E add yaw to the translation in "
                      "effect; press '{}' to stop capture and zero velocity references, space to clear the axes",
                      keyboardExitKey[0]);
  }
  registerCommandGUI();
  mc_solver::RollingContactConstraintOptions options;
  options.terrainNormal = settings("terrainNormal", Eigen::Vector3d{0.0, 0.0, 1.0});
  if(!options.terrainNormal.allFinite() || options.terrainNormal.norm() < 1e-8)
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>("RollingContact terrainNormal must be finite and nonzero");
  }
  terrainNormal_ = options.terrainNormal.normalized();
  // One normalised vector for both constraints, and the same one every cycle.
  // RollingContactDynamicsConstraint::terrainNormal() normalises its argument
  // internally while RollingContactConstraint::terrainNormal() stores it
  // verbatim, so feeding the two setters the raw configured vector would leave
  // them holding different normals whenever the configuration is not already a
  // unit vector. Normalise once here and hand terrainNormal_ to both.
  options.terrainNormal = terrainNormal_;
  // This basis is the world-fixed yaw reference frame, NOT the chassis-aligned
  // planar basis of assumption A1. measuredYaw is
  // atan2(heading . terrainTangentY_, heading . terrainTangentX_), so a basis
  // that rotated with the chassis would make it identically zero and silently
  // break every piece of yaw bookkeeping downstream. The chassis-aligned basis
  // A1 asks for is where rho_i is resolved, and wheelOffsets_ above already
  // does that; see mc_rbdyn::planarContactBasis().
  Eigen::Vector3d terrainForward = Eigen::Vector3d::UnitX();
  if((terrainForward - terrainNormal_ * terrainNormal_.x()).norm() < 1e-8)
  {
    terrainForward = Eigen::Vector3d::UnitY();
  }
  // planarContactBasis() throws when even the fallback axis is parallel to the
  // normal, instead of normalizing what is left of it.
  const Eigen::Matrix3d terrainBasis = mc_rbdyn::planarContactBasis(terrainNormal_, terrainForward);
  terrainTangentX_ = terrainBasis.col(0);
  terrainTangentY_ = terrainBasis.col(1);
  options.longitudinal = longitudinalMode(settings);
  options.velocityGain = settings("velocityGain", 20.0);
  options.rollingWeight = rollingWeight_;
  options.constrainNormal = true;
  options.differentialPlanar = !fourSteering_;
  // Four lateral rows on three planar chassis DOF are over-determined: at
  // steering angles that share no common ICR the block has full row rank 3, so
  // hard rows admit only the zero twist. Softening all four keeps the QP
  // feasible and makes the incompatibility observable as lateralSlack(). A
  // two-wheel differential chassis is not over-determined and keeps hard rows.
  options.softLateralRows = fourSteering_;
  if(fourSteering_)
  {
    // Swept, not guessed, and bounded from BOTH sides - the value below sits in
    // a window about two decades wide, not on a monotone "bigger is safer"
    // curve. A simple comparison against the tracking weights does not predict
    // either edge, because the lateral rows contend for the chassis DOF with
    // the predicted-rate rows, which act on the wheel DOF instead.
    //
    // Too small and a commanded-twist step skids: the worst lateral slip of
    // FourSteeringCommandChangeKeepsResidualsBounded is 1.85 m/s at 1e5, 1.58
    // at 2e6 and 1.57 at 3e6 - all far outside its 0.15 m/s bound - then falls
    // off a cliff to 0.082 at 5e6, 0.040 at 1e7 and 0.025 at 3e7.
    //
    // Too large and the lateral rows resist the chassis yaw they are supposed
    // only to keep honest, which the kinematic ticker cannot see at all. In
    // MuJoCo (four-pure-yaw, Tasks) the maximum yaw tracking error rises
    // monotonically with the weight - 0.064 rad at 1e6, 0.089 at 1e7, 0.131 at
    // 1e8 - and breaks the suite's 0.15 rad bound at 1e9 (0.171) and 1e10
    // (0.202). The hard-row implementation this replaces sat at 0.060.
    //
    // 1e7 is the balance point: 3.3x in weight above the skid cliff with a 3.7x
    // margin in the metric, and 1.7x below the yaw bound, with a MuJoCo lateral
    // slip and drive torque both better than the hard-row baseline.
    options.lateralSlackWeight = settings("lateralSlackWeight", 1e7);
    if(!std::isfinite(options.lateralSlackWeight) || options.lateralSlackWeight <= 0.0)
    {
      mc_rtc::log::error_and_throw<std::invalid_argument>(
          "RollingContact lateralSlackWeight must be finite and positive");
    }
    options.trackRotatingRates = true;
    // A rate row predicts rate^+ = rate + dt * S * alphaD, so its coefficients
    // carry dt and its weight enters the objective multiplied by dt^2. Express
    // the defaults as the equivalent acceleration-task weight (1000, twice the
    // chassis tasks) divided by dt^2; the documented option keeps its own
    // "weight on the rate residual" meaning. With a plain weight of 200 the
    // rate rows contribute 200 * dt^2 = 5e-3 and the posture task, at 100,
    // silently keeps ownership of the wheel degrees of freedom.
    const double rateWeightScale = 1.0 / (dt * dt);
    options.rollingRateWeight = settings("rollingRateWeight", 1000.0 * rateWeightScale);
    options.steeringRateWeight = settings("steeringRateWeight", 1000.0 * rateWeightScale);
  }
  // Tikhonov weight on the contact-force generators, forwarded to
  // RollingContactDynamicsConstraint (which keeps its own default of 0.0; see
  // that class for why the Tasks backend does not need it for conditioning).
  //
  // It is exposed here because it is the one knob that removes the *other*
  // consequence of the four-contact normal-force null space documented in
  // updateModes(): the QP is free to choose any point in it, and the point it
  // lands on is a physically wrong load split. Swept against real mc_mujoco on
  // the Ranger, four-steering crab at 0.2 m/s, measuring the steady normal
  // forces and the achieved body speed over t >= 9 s:
  //
  //   eps  = 0.00  ->  N = [171, 0, 200, 365] N
  //   eps  = 0.05  ->  N = [203, 0, 147, 385] N,  speed 0.131 / 0.200
  //   eps  = 0.20  ->  N = [177, 181, 187, 190] N, speed 0.1976 / 0.200
  //   eps  = 0.50  ->  N = [177, 181, 187, 190] N, speed 0.1976 / 0.200
  //   eps  = 1.00  ->  N = [177, 181, 187, 190] N, speed 0.1976 / 0.200
  //
  // The knee is between 0.05 and 0.2 and the metrics are flat above it: the
  // penalised subspace produces no net wrench, so picking the minimum-norm
  // point inside it costs the tracking tasks nothing measurable. It is not the
  // default because it does not by itself keep every manoeuvre attached -
  // four-steering ackermann_left still drives one wheel's multiplier onto its
  // lambda >= 0 bound at eps = 0.5 - and because a non-zero value has a
  // measured bias on the solved contact force (see commit 1f136644ad).
  const double generatorRegularization = settings("generatorRegularization", 0.0);
  dynamics_ = std::make_unique<mc_solver::RollingContactDynamicsConstraint>(robots(), 0, dt, wheels_, false,
                                                                            generatorRegularization);
  dynamics_->terrainNormal(terrainNormal_);
  rolling_ = std::make_unique<mc_solver::RollingContactConstraint>(robots(), 0, wheels_, options);

  mc_rbdyn::RollingContactModeThresholds thresholds;
  thresholds.slipEnter = settings("slipEnter", 0.05);
  thresholds.slipExit = settings("slipExit", 0.02);
  thresholds.residualEnter = settings("residualEnter", 0.05);
  thresholds.residualExit = settings("residualExit", 0.02);
  thresholds.normalForceEnter = settings("normalForceEnter", 5.0);
  thresholds.normalForceExit = settings("normalForceExit", 1.0);
  thresholds.frictionMarginEnter = settings("frictionMarginEnter", 1e-3);
  thresholds.frictionMarginExit = settings("frictionMarginExit", -1e-5);
  thresholds.torqueMarginEnter = settings("torqueMarginEnter", 0.1);
  thresholds.torqueMarginExit = settings("torqueMarginExit", 0.0);
  thresholds.minimumDwell = settings("minimumDwell", 0.1);
  thresholds.transitionTime = settings("transitionTime", 0.2);
  thresholds.filterTimeConstant = settings("filterTimeConstant", 0.02);
  thresholds.validate();
  modeManagers_.reserve(wheels_.size());
  for(const auto & wheel : wheels_)
  {
    modeManagers_.emplace_back(thresholds);
    modeManagers_.back().reset(wheel.mode, wheel.mode,
                               wheel.mode == mc_rbdyn::RollingContactMode::Detached ? 0.0 : 1.0);
  }

  solver().addConstraintSet(*dynamics_);
  solver().addConstraintSet(*rolling_);
  postureTask->stiffness(settings("postureStiffness", 5.0));
  postureTask->weight(settings("postureWeight", 100.0));
  if(fourSteering_)
  {
    const double steeringStiffness = settings("steeringStiffness", 1000.0);
    // The QP's steering-rate rows now own the hinge motion. Keep the posture
    // task on these joints only as a weak regulariser so it cannot fight them.
    const double steeringWeight = settings("steeringPostureWeight", 1.0);
    if(!std::isfinite(steeringStiffness) || steeringStiffness <= 0.0 || !std::isfinite(steeringWeight)
       || steeringWeight <= 0.0)
    {
      mc_rtc::log::error_and_throw<std::invalid_argument>(
          "RollingContact steeringStiffness and steeringPostureWeight must be finite and positive");
    }
    std::vector<tasks::qp::JointStiffness> steeringGains;
    std::map<std::string, double> steeringWeights;
    steeringGains.reserve(wheels_.size());
    for(const auto & wheel : wheels_)
    {
      steeringGains.emplace_back(wheel.steeringJoint, steeringStiffness);
      steeringWeights.emplace(wheel.steeringJoint, steeringWeight);
    }
    postureTask->jointStiffness(solver(), steeringGains);
    postureTask->jointWeights(steeringWeights);
  }
  solver().addTask(postureTask);
  // eq:planar-twist-weight: the planar twist error [vx, vy, omega] is
  // weighted by a diagonal matrix, not a scalar, because vx/vy (m/s) and
  // omega (rad/s) are not otherwise comparable. See twistWeight_'s
  // declaration for the unit convention; the rows outside the planar twist
  // (basePositionTask_'s z, baseOrientationTask_'s x/y) are not part of
  // eq:planar-twist-weight and always keep weight 1.
  twistWeight_ = readTwistWeight(settings);
  if(!twistWeight_.allFinite() || (twistWeight_.array() <= 0.0).any())
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>("RollingContact twistWeight must be finite and positive");
  }
  baseOrientationTask_ = std::make_shared<mc_tasks::OrientationTask>(
      "chassis", robots(), 0, settings("baseOrientationStiffness", 10.0),
      settings("baseOrientationWeight", 500.0));
  baseOrientationTask_->dimWeight(Eigen::Vector3d{1.0, 1.0, twistWeight_.z()});
  solver().addTask(baseOrientationTask_);
  basePositionTask_ = std::make_shared<mc_tasks::PositionTask>(
      "chassis", robots(), 0, settings("basePositionStiffness", 5.0), settings("basePositionWeight", 500.0));
  basePositionTask_->dimWeight(Eigen::Vector3d{twistWeight_.x(), twistWeight_.y(), 1.0});
  solver().addTask(basePositionTask_);
  solver().setContacts({});
  driveTargets_.resize(wheels_.size(), 0.0);
  wheelReferenceRates_.resize(wheels_.size(), 0.0);
  steeringTargets_.resize(wheels_.size(), 0.0);
  steeringRateReferences_.resize(wheels_.size(), 0.0);
  rollingResiduals_.resize(wheels_.size(), 0.0);
  lateralResiduals_.resize(wheels_.size(), 0.0);
  normalResiduals_.resize(wheels_.size(), 0.0);
  accelerationResiduals_.resize(wheels_.size(), 0.0);
  lateralAccelerationResiduals_.resize(wheels_.size(), 0.0);
  normalAccelerationResiduals_.resize(wheels_.size(), 0.0);
  hardPromotionResiduals_.resize(wheels_.size(), 0.0);
  normalForces_.resize(wheels_.size(), 0.0);
  tangentialForces_.resize(wheels_.size(), 0.0);
  frictionMargins_.resize(wheels_.size(), 0.0);
  driveTorques_.resize(wheels_.size(), 0.0);
  driveTorqueMargins_.resize(wheels_.size(), 0.0);
  appliedActivations_.resize(wheels_.size(), 1.0);
  hardPromotionPending_.resize(wheels_.size(), false);
  externalMeasurements_.resize(wheels_.size());
  measuredDrivePositions_.resize(wheels_.size(), 0.0);
  keyboardRefVel_.setZero(robot().mb().nrDof());
  alphaDBuffer_.setZero(robot().mb().nrDof());
  odometryWheels_.resize(wheels_.size());
  odometryRates_.setZero(static_cast<Eigen::Index>(wheels_.size()));
  odometryMatrix_.setZero(2 * static_cast<Eigen::Index>(wheels_.size()), 3);
  odometryRhs_.setZero(2 * static_cast<Eigen::Index>(wheels_.size()));
  // Install the posture-target keys once. run() only ever rewrites the single
  // value behind each key, so no node and no value vector is ever reallocated.
  for(const auto & wheel : wheels_)
  {
    postureTargets_.emplace(wheel.driveJoint, std::vector<double>{0.0});
    if(!wheel.steeringJoint.empty()) { postureTargets_.emplace(wheel.steeringJoint, std::vector<double>{0.0}); }
  }

  registerDatastoreCalls();

  registerLogEntries();
  registerStatusGUI();
  registerAttitudeGUI();
  mc_rtc::log::success("RollingContact CPU controller initialized for {} ({})", robot().name(), scenario_);
}

void MCRollingContactController::registerCommandGUI()
{
  gui()->addElement(
      {"Rolling Contact", "Command"},
      // Outside the keyboard scenario the GUI is the operator's only entry
      // point, so it publishes the commanded twist directly. The keyboard poll
      // republishes the key state plus these offsets every cycle instead.
      mc_rtc::gui::NumberInput(
          "Forward velocity", [this]() { return guiForwardCommand_; },
          [this](double value)
          {
            if(!std::isfinite(value)) { return; }
            guiForwardCommand_ = value;
            if(scenario_ != "keyboard") { setCommandedTwist({value, commandedTwist_.y(), commandedTwist_.z()}); }
          }),
      mc_rtc::gui::NumberInput(
          "Lateral velocity", [this]() { return guiLateralCommand_; },
          [this](double value)
          {
            if(!std::isfinite(value)) { return; }
            guiLateralCommand_ = value;
            if(scenario_ != "keyboard") { setCommandedTwist({commandedTwist_.x(), value, commandedTwist_.z()}); }
          }),
      mc_rtc::gui::NumberInput(
          "Yaw velocity", [this]() { return guiYawCommand_; },
          [this](double value)
          {
            if(!std::isfinite(value)) { return; }
            guiYawCommand_ = value;
            if(scenario_ != "keyboard") { setCommandedTwist({commandedTwist_.x(), commandedTwist_.y(), value}); }
          }),
      mc_rtc::gui::Button(
          "Clear command", [this]()
          {
            guiForwardCommand_ = 0.0;
            guiLateralCommand_ = 0.0;
            guiYawCommand_ = 0.0;
            if(scenario_ != "keyboard") { setCommandedTwist(Eigen::Vector3d::Zero()); }
            if(keyboard_) { keyboard_->clear(); }
          }));
}

void MCRollingContactController::registerDatastoreCalls()
{
  datastore().make_call(
      "RollingContact::SetMeasuredContact",
      [this](const std::string & name,
             double rollingSlip,
             double lateralSlip,
             double normalForce,
             double tangentialForce,
             bool valid)
      {
        auto & measurement = externalMeasurements_[wheelIndex(name, "measurement")];
        measurement.rollingSlip = rollingSlip;
        measurement.lateralSlip = lateralSlip;
        measurement.normalForce = normalForce;
        measurement.tangentialForce = tangentialForce;
        measurement.age = 0.0;
        measurement.valid = valid && std::isfinite(rollingSlip) && std::isfinite(lateralSlip)
                            && std::isfinite(normalForce) && normalForce >= 0.0 && std::isfinite(tangentialForce)
                            && tangentialForce >= 0.0;
      });
  datastore().make_call("RollingContact::GetEstimatedMode",
                        [this](const std::string & name) -> std::string
                        {
                          return mc_rbdyn::to_string(modeManagers_[wheelIndex(name, "mode")].state().estimated);
                        });
  datastore().make_call("RollingContact::GetActivation",
                        [this](const std::string & name) -> double
                        { return modeManagers_[wheelIndex(name, "activation")].state().activation; });
  makeWheelValueCall("RollingContact::GetSolverActivation", "solver activation", appliedActivations_);
  makeWheelValueCall("RollingContact::GetAccelerationResidual", "acceleration residual", accelerationResiduals_);
  makeWheelValueCall("RollingContact::GetHardPromotionResidual", "hard promotion residual", hardPromotionResiduals_);
  makeWheelValueCall("RollingContact::GetLateralAccelerationResidual", "lateral acceleration",
                     lateralAccelerationResiduals_);
  makeWheelValueCall("RollingContact::GetNormalAccelerationResidual", "normal acceleration",
                     normalAccelerationResiduals_);
  datastore().make_call("RollingContact::GetBasePositionTarget", [this]() { return basePositionTarget_; });
  datastore().make_call("RollingContact::GetBaseYawTarget", [this]() { return baseYawTarget_; });
  datastore().make_call("RollingContact::GetKeyboardYawTarget", [this]() { return keyboardYawTarget_; });
  datastore().make_call("RollingContact::GetKeyboardYawError", [this]() { return keyboardYawError_; });
  datastore().make_call("RollingContact::GetKeyboardYawCorrection", [this]() { return keyboardYawCorrection_; });
  datastore().make_call("RollingContact::GetKeyboardStatus", [this]()
                        { return keyboard_ ? keyboard_->status() : std::string{"disabled"}; });
  datastore().make_call("RollingContact::GetContactFallback", [this]() { return contactFallback_; });
  datastore().make_call("RollingContact::GetRollingWeight", [this]() { return rolling_->rollingWeight(); });
  datastore().make_call("RollingContact::GetDynamicsResidual", [this]() { return dynamicsResidual_; });
  datastore().make_call("RollingContact::GetFloatingBaseEffortNorm", [this]() { return floatingBaseEffortNorm_; });
  makeWheelValueCall("RollingContact::GetQPNormalForce", "QP-force", normalForces_);
  makeWheelValueCall("RollingContact::GetQPTangentialForce", "QP-force", tangentialForces_);
  makeWheelValueCall("RollingContact::GetDriveTarget", "drive target", driveTargets_);
  datastore().make_call("RollingContact::GetDriveAcceleration",
                        [this](const std::string & name) -> double
                        {
                          const auto & wheel = wheels_[wheelIndex(name, "drive acceleration")];
                          return robot().mbc().alphaD[robot().jointIndexByName(wheel.driveJoint)][0];
                        });
  datastore().make_call("RollingContact::GetDrivePosition",
                        [this](const std::string & name) -> double
                        {
                          const auto & wheel = wheels_[wheelIndex(name, "drive position")];
                          return robot().mbc().q[robot().jointIndexByName(wheel.driveJoint)][0];
                        });
  datastore().make_call("RollingContact::GetDriveVelocity",
                        [this](const std::string & name) -> double
                        {
                          const auto & wheel = wheels_[wheelIndex(name, "drive velocity")];
                          return robot().mbc().alpha[robot().jointIndexByName(wheel.driveJoint)][0];
                        });
  datastore().make_call("RollingContact::GetBackend", [this]()
                        { return solver().backend() == Backend::Tasks ? std::string{"Tasks"} : std::string{"TVM"}; });
  datastore().make_call("RollingContact::GetPostureEvalNorm", [this]() { return postureTask->eval().norm(); });
  datastore().make_call("RollingContact::GetPositionEvalNorm", [this]() { return basePositionTask_->eval().norm(); });
  datastore().make_call("RollingContact::GetOrientationEvalNorm",
                        [this]() { return baseOrientationTask_->eval().norm(); });
  // Direct introspection of the per-axis twist weight (eq:planar-twist-weight,
  // R4/QP-06): basePositionTask_'s [x, y, z] and baseOrientationTask_'s
  // [roll, pitch, yaw] dimWeight, exactly as fed to the QP by
  // SetPointTaskCommon::computeQC (Tasks/src/QPTasks.cpp:67-70). Reading this
  // back is a direct, deterministic check that twistWeight_ reached the two
  // tasks' rows as documented, independent of whatever the closed-loop
  // trajectory happens to do with it.
  datastore().make_call("RollingContact::GetBasePositionDimWeight",
                        [this]() -> Eigen::Vector3d { return basePositionTask_->dimWeight(); });
  datastore().make_call("RollingContact::GetBaseOrientationDimWeight",
                        [this]() -> Eigen::Vector3d { return baseOrientationTask_->dimWeight(); });
  datastore().make_call("RollingContact::GetOdometryTwist", [this]() { return odometryTwist_; });
  datastore().make_call("RollingContact::GetOdometryResidual", [this]() { return odometryResidual_; });
  // Published under the VelocityAidedTilt namespace, not RollingContact::,
  // because the consumer is an observer that resolves the key by name and knows
  // nothing about this controller - the same arrangement
  // KinematicInertialPoseObserver has with "KinematicAnchorFrame::<robot>".
  // Taking the robot as an argument (rather than closing over realRobot())
  // follows that precedent too, and lets a caller ask the same question of the
  // control robot.
  datastore().make_call("VelocityAidedTilt::SensorVelocity::" + robot().name(),
                        [this](const mc_rbdyn::Robot & measured) -> Eigen::Vector3d
                        {
                          double residual = 0.0;
                          const Eigen::Vector3d twist = wheelOdometryTwist(measured, residual);
                          // A LINEAR VELOCITY, not the planar twist: the third
                          // component of that twist is a yaw rate, and handing
                          // it over as if it were vz would feed the consumer a
                          // rad/s in a m/s slot. The consumer gets its angular
                          // velocity from the gyroscope. vz is zero by the
                          // rolling constraint's own normal row.
                          return Eigen::Vector3d{twist.x(), twist.y(), 0.0};
                        });
  // How much wheel support that twist actually rests on, in wheels: the sum of
  // the activations the rows above were weighted by. A consumer needs it to
  // tell "the chassis is standing still" from "nothing is touching the ground",
  // which the twist alone cannot express - both are the zero vector.
  datastore().make_call("VelocityAidedTilt::VelocityActivation::" + robot().name(), [this]() -> double
                        { return std::accumulate(appliedActivations_.begin(), appliedActivations_.end(), 0.0); });
  datastore().make_call("RollingContact::GetHardRhsNorm", [this]() { return rolling_->hardRhs().norm(); });
  datastore().make_call("RollingContact::GetSlidingGenerator",
                        [this](const std::string & name) { return dynamics_->slidingGenerator(name); });
}

void MCRollingContactController::registerLogEntries()
{
  logger().addLogEntry("RollingContact_scenario", [this]() { return scenario_; });
  logger().addLogEntry("RollingContact_backend", [this]()
                       { return solver().backend() == Backend::Tasks ? std::string{"Tasks"} : std::string{"TVM"}; });
  logger().addLogEntry("RollingContact_solver_success", [this]() { return lastSolverSuccess_; });
  logger().addLogEntry("RollingContact_closed_loop_feedback", [this]() { return closedLoopFeedback_; });
  logger().addLogEntry("RollingContact_contact_fallback", [this]() { return contactFallback_; });
  logger().addLogEntry("RollingContact_rolling_weight", [this]() { return rolling_->rollingWeight(); });
  logger().addLogEntry("RollingContact_dynamics_residual", [this]() { return dynamicsResidual_; });
  logger().addLogEntry("RollingContact_odometry_twist",
                       [this]() -> const Eigen::Vector3d & { return odometryTwist_; });
  logger().addLogEntry("RollingContact_odometry_residual", [this]() { return odometryResidual_; });
  // Radians, both of them, in mc_rbdyn::rpyFromMat's convention; the GUI is
  // where degrees appear. "reference" and not "ground truth": it is the
  // FloatingBase body sensor's attitude, which is MuJoCo's qpos under
  // mc_mujoco but the control robot's own integrated pose under mc_rtc_ticker.
  logger().addLogEntry("RollingContact_attitude_estimated_rp",
                       [this]() -> const Eigen::Vector2d & { return attitudeMonitorEstimatedRP_; });
  logger().addLogEntry("RollingContact_attitude_reference_rp",
                       [this]() -> const Eigen::Vector2d & { return attitudeMonitorReferenceRP_; });
  logger().addLogEntry("RollingContact_attitude_error", [this]() { return attitudeMonitorError_; });
  logger().addLogEntry("RollingContact_attitude_valid", [this]() { return attitudeMonitorValid_; });
  logger().addLogEntry("RollingContact_floating_base_effort_norm", [this]() { return floatingBaseEffortNorm_; });
  logger().addLogEntry("RollingContact_update_ms", [this]() { return updateTimeMs_; });
  logger().addLogEntry("RollingContact_solve_and_build_ms", [this]() { return solveAndBuildTimeMs_; });
  logger().addLogEntry("RollingContact_total_ms", [this]() { return totalTimeMs_; });
  logger().addLogEntry("RollingContact_max_longitudinal_residual", [this]() { return maxRollingResidual_; });
  logger().addLogEntry("RollingContact_max_lateral_residual", [this]() { return maxLateralResidual_; });
  logger().addLogEntry("RollingContact_lateral_slack_norm", [this]() { return lateralSlackNorm_; });
  logger().addLogEntry("RollingContact_min_friction_margin", [this]() { return minFrictionMargin_; });
  logger().addLogEntry("RollingContact_diagnostics_valid", [this]() { return diagnosticsValid_; });
  logger().addLogEntry("RollingContact_invalid_reason", [this]() { return invalidReason_; });
  logger().addLogEntry("RollingContact_reference_linear_speed", [this]() { return referenceLinearSpeed_; });
  logger().addLogEntry("RollingContact_reference_lateral_speed", [this]() { return referenceLateralSpeed_; });
  logger().addLogEntry("RollingContact_reference_yaw_rate", [this]() { return referenceYawRate_; });
  logger().addLogEntry("RollingContact_gui_forward_command", [this]() { return guiForwardCommand_; });
  logger().addLogEntry("RollingContact_gui_lateral_command", [this]() { return guiLateralCommand_; });
  logger().addLogEntry("RollingContact_gui_yaw_command", [this]() { return guiYawCommand_; });
  logger().addLogEntry("RollingContact_keyboard_status", [this]()
                       { return keyboard_ ? keyboard_->status() : std::string{"disabled"}; });
  logger().addLogEntry("RollingContact_keyboard_running", [this]() { return keyboard_ && keyboard_->running(); });
  logger().addLogEntry("RollingContact_commandedTwist", [this]() -> const Eigen::Vector3d & { return commandedTwist_; });
  logger().addLogEntry("RollingContact_keyboard_yaw_error", [this]() { return keyboardYawError_; });
  logger().addLogEntry("RollingContact_keyboard_yaw_correction", [this]() { return keyboardYawCorrection_; });
  // The two chassis tasks' raw errors. The orientation one in particular is a
  // rotation vector: it is what saturates at pi when a target drifts to the
  // antipode of the measured pose, and neither task publishes its own eval to
  // the log, so a post-mortem cannot otherwise tell a healthy run from one that
  // is a few cycles away from that singularity.
  logger().addLogEntry("RollingContact_base_orientation_eval",
                       [this]() -> Eigen::Vector3d { return baseOrientationTask_->eval(); });
  logger().addLogEntry("RollingContact_base_position_eval",
                       [this]() -> Eigen::Vector3d { return basePositionTask_->eval(); });
  logger().addLogEntry("RollingContact_base_position_target", [this]() { return basePositionTarget_; });
  logger().addLogEntry("RollingContact_base_yaw_target", [this]() { return baseYawTarget_; });
  logger().addLogEntry("RollingContact_base_position_reference_velocity", [this]()
                       { return baseReferenceVelocity_; });
  logger().addLogEntry("RollingContact_base_tracking_velocity", [this]() { return baseTrackingVelocity_; });
  logger().addLogEntry("RollingContact_base_angular_reference_velocity", [this]()
                       { return baseReferenceAngularVelocity_; });
  logger().addLogEntry("RollingContact_base_position_tracking_error", [this]()
                       {
                         const auto & state = closedLoopFeedback_ ? realRobot() : robot();
                         return (basePositionTarget_ - state.posW().translation()).norm();
                       });
  // In closed-loop operation robot() is the controller/output state and may
  // contain the one-step prediction produced by the QP. realRobot() is the
  // measured state and is what an interface/diagnostic consumer expects for a
  // floating-base pose/twist. Keep the open-loop behavior unchanged.
  logger().addLogEntry("RollingContact_base_pose", [this]() -> const sva::PTransformd &
                       { return closedLoopFeedback_ ? realRobot().posW() : robot().posW(); });
  logger().addLogEntry("RollingContact_base_twist", [this]() -> const sva::MotionVecd &
                       { return closedLoopFeedback_ ? realRobot().velW() : robot().velW(); });
  // The FloatingBase packet and the chassis frame must describe the same
  // physical origin. Compare the measured robot (not the one-step predicted
  // control state) so this remains a meaningful runtime model check. The raw
  // BodySensor packet itself only ever lands on robot(): the observer
  // pipeline consumes it to estimate realRobot()'s posW()/velW(), but does
  // not copy the sensor object into realRobot(), so read it from robot()
  // unconditionally and only switch the comparison target (the frame) with
  // closedLoopFeedback_.
  logger().addLogEntry("RollingContact_floating_base_chassis_error", [this]()
                       {
                         if(!robot().hasBodySensor("FloatingBase"))
                         {
                           return std::numeric_limits<double>::infinity();
                         }
                         const auto & sensor = robot().bodySensor("FloatingBase");
                         const sva::PTransformd sensorPose(sensor.orientation(), sensor.position());
                         const auto & measured = closedLoopFeedback_ ? realRobot() : robot();
                         return (sensorPose.matrix() - measured.frame("chassis").position().matrix()).norm();
                       });
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    const auto & wheel = wheels_[i];
    const auto joint = robot().jointIndexByName(wheel.driveJoint);
    logger().addLogEntry("RollingContact_" + wheel.name + "_position",
                         [this, joint]() { return robot().mbc().q[joint][0]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_rate",
                         [this, joint]() { return robot().mbc().alpha[joint][0]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_measured_rate",
                         [this, joint]()
                         {
                           const auto & refJointOrder = robot().refJointOrder();
                           const auto it = std::find(refJointOrder.begin(), refJointOrder.end(),
                                                     robot().mb().joint(joint).name());
                           const auto & velocities = robot().encoderVelocities();
                           if(it != refJointOrder.end())
                           {
                             const auto index = static_cast<size_t>(std::distance(refJointOrder.begin(), it));
                             if(index < velocities.size() && std::isfinite(velocities[index]))
                             {
                               return velocities[index];
                             }
                           }
                           return robot().mbc().alpha[joint][0];
                         });
    logger().addLogEntry("RollingContact_" + wheel.name + "_acceleration",
                         [this, joint]() { return robot().mbc().alphaD[joint][0]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_target", [this, i]() { return driveTargets_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_rollingRateRef",
                         [this, i]() { return wheelReferenceRates_[i]; });
    if(!wheel.steeringJoint.empty())
    {
      const auto steering = robot().jointIndexByName(wheel.steeringJoint);
      logger().addLogEntry("RollingContact_" + wheel.name + "_steering_position",
                           [this, steering]() { return robot().mbc().q[steering][0]; });
      logger().addLogEntry("RollingContact_" + wheel.name + "_steering_target",
                           [this, i]() { return steeringTargets_[i]; });
      logger().addLogEntry("RollingContact_" + wheel.name + "_steering_rate",
                           [this, steering]() { return robot().mbc().alpha[steering][0]; });
      logger().addLogEntry("RollingContact_" + wheel.name + "_steeringRateRef",
                           [this, i]() { return steeringRateReferences_[i]; });
    }
    logger().addLogEntry("RollingContact_" + wheel.name + "_requested_mode", [this, i]()
                         { return std::string{mc_rbdyn::to_string(modeManagers_[i].state().requested)}; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_mode", [this, i]()
                         { return std::string{mc_rbdyn::to_string(modeManagers_[i].state().estimated)}; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_activation",
                         [this, i]() { return modeManagers_[i].state().activation; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_solver_activation",
                         [this, i]() { return appliedActivations_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_filtered_slip",
                         [this, i]() { return modeManagers_[i].state().filteredSlip; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_dwell_time",
                         [this, i]() { return modeManagers_[i].state().dwellTime; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_measurement_valid",
                         [this, i]() { return modeManagers_[i].state().measurementValid; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_invalid_reason",
                         [this, i]() { return modeManagers_[i].state().invalidReason; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_rolling_residual",
                         [this, i]() { return rollingResiduals_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_lateral_residual",
                         [this, i]() { return lateralResiduals_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_normal_residual",
                         [this, i]() { return normalResiduals_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_acceleration_residual",
                         [this, i]() { return accelerationResiduals_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_lateral_acceleration_residual",
                         [this, i]() { return lateralAccelerationResiduals_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_normal_acceleration_residual",
                         [this, i]() { return normalAccelerationResiduals_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_hard_promotion_residual",
                         [this, i]() { return hardPromotionResiduals_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_normal_force",
                         [this, i]() { return normalForces_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_tangential_force",
                         [this, i]() { return tangentialForces_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_friction_margin",
                         [this, i]() { return frictionMargins_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_drive_torque",
                         [this, i]() { return driveTorques_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_drive_torque_margin",
                         [this, i]() { return driveTorqueMargins_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_sliding_generator",
                         [this, i]() { return dynamics_->slidingGenerator(wheels_[i].name); });
    logger().addLogEntry("RollingContact_" + wheel.name + "_simulator_measurement_valid",
                         [this, i]()
                         {
                           return externalMeasurements_[i].valid
                                  && externalMeasurements_[i].age <= 2.0 * solver().dt();
                         });
    logger().addLogEntry("RollingContact_" + wheel.name + "_measured_rolling_slip",
                         [this, i]() { return externalMeasurements_[i].rollingSlip; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_measured_lateral_slip",
                         [this, i]() { return externalMeasurements_[i].lateralSlip; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_measured_normal_force",
                         [this, i]() { return externalMeasurements_[i].normalForce; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_measured_tangential_force",
                         [this, i]() { return externalMeasurements_[i].tangentialForce; });
    if(!wheel.steeringJoint.empty())
    {
      const auto steeringJoint = robot().jointIndexByName(wheel.steeringJoint);
      logger().addLogEntry("RollingContact_" + wheel.name + "_measured_steering_position",
                           [this, steeringJoint]()
                           {
                             // The MuJoCo adapter publishes encoder values on the
                             // controlled robot. Use that raw stream directly for the
                             // measured steering angle rather than realRobot(): the
                             // Encoder observer only updates realRobot() when
                             // closedLoopFeedback_ has an ObserverPipelines block
                             // configured and run, and this log entry should stay
                             // meaningful regardless of that.
                             const auto & refJointOrder = robot().refJointOrder();
                             const auto it = std::find(refJointOrder.begin(), refJointOrder.end(),
                                                       robot().mb().joint(steeringJoint).name());
                             const auto & encoders = robot().encoderValues();
                             if(it != refJointOrder.end())
                             {
                               const auto index = static_cast<size_t>(std::distance(refJointOrder.begin(), it));
                               if(index < encoders.size() && std::isfinite(encoders[index])) { return encoders[index]; }
                             }
                             return robot().mbc().q[steeringJoint][0];
                           });
    }
  }
}

void MCRollingContactController::registerStatusGUI()
{
  gui()->addElement({"Rolling Contact"},
                    mc_rtc::gui::Label("Scenario", [this]() { return scenario_; }),
                    mc_rtc::gui::Label("Backend", [this]()
                                       { return solver().backend() == Backend::Tasks ? "Tasks" : "TVM"; }),
                    mc_rtc::gui::Label("Diagnostics valid", [this]() { return diagnosticsValid_; }),
                    mc_rtc::gui::Label("Invalid reason", [this]() { return invalidReason_; }));
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    gui()->addElement(
        {"Rolling Contact", wheels_[i].name},
        mc_rtc::gui::Label("Requested mode", [this, i]()
                           { return std::string{mc_rbdyn::to_string(modeManagers_[i].state().requested)}; }),
        mc_rtc::gui::Label("Estimated mode", [this, i]()
                           { return std::string{mc_rbdyn::to_string(modeManagers_[i].state().estimated)}; }),
        mc_rtc::gui::Label("Activation", [this, i]() { return modeManagers_[i].state().activation; }),
        mc_rtc::gui::Label("Solver activation", [this, i]() { return appliedActivations_[i]; }),
        mc_rtc::gui::Label("Slip", [this, i]() { return modeManagers_[i].state().filteredSlip; }),
        mc_rtc::gui::Label("Normal force", [this, i]() { return normalForces_[i]; }),
        mc_rtc::gui::Label("Friction margin", [this, i]() { return frictionMargins_[i]; }),
        mc_rtc::gui::Label("Drive torque margin", [this, i]() { return driveTorqueMargins_[i]; }));
  }
}

void MCRollingContactController::registerAttitudeGUI()
{
  using namespace mc_rtc::gui;
  // static, so the arrow lambdas below can use it without capturing it.
  static constexpr double kArrowLength = 0.6;
  // Numbers first: the pair of roll/pitch readings and the single angle between
  // the two normals that they decompose. Degrees here and only here - the log
  // and the members stay in radians.
  gui()->addElement({"Rolling Contact", "Attitude"},
                    Label("Estimate available", [this]() { return attitudeMonitorValid_; }),
                    ArrayLabel("Estimated roll/pitch [deg]", {"roll", "pitch"},
                               [this]() -> Eigen::Vector2d
                               {
                                 return {mc_rtc::constants::toDeg(attitudeMonitorEstimatedRP_.x()),
                                         mc_rtc::constants::toDeg(attitudeMonitorEstimatedRP_.y())};
                               }),
                    ArrayLabel("Reference roll/pitch [deg]", {"roll", "pitch"},
                               [this]() -> Eigen::Vector2d
                               {
                                 return {mc_rtc::constants::toDeg(attitudeMonitorReferenceRP_.x()),
                                         mc_rtc::constants::toDeg(attitudeMonitorReferenceRP_.y())};
                               }),
                    Label("Attitude error [deg]",
                          [this]() { return mc_rtc::constants::toDeg(attitudeMonitorError_); }));
  // Then the same thing in the 3D view, where a degree of tilt is easier to see
  // than to read: two arrows out of the chassis origin, green for the estimate
  // and grey for the reference. They coincide exactly when the estimate is
  // right, which is the whole point - and they still coincide when no observer
  // is running, which the "Estimate available" label above is there to
  // distinguish.
  gui()->addElement({"Rolling Contact", "Attitude"},
                    Arrow(
                        "Estimated normal", ArrowConfig{Color::Green},
                        [this]() -> Eigen::Vector3d { return realRobot().posW().translation(); },
                        [this]() -> Eigen::Vector3d
                        { return realRobot().posW().translation() + kArrowLength * attitudeMonitorEstimatedNormal_; }),
                    Arrow(
                        "Reference normal", ArrowConfig{Color::Gray},
                        [this]() -> Eigen::Vector3d { return realRobot().posW().translation(); },
                        [this]() -> Eigen::Vector3d
                        { return realRobot().posW().translation() + kArrowLength * attitudeMonitorReferenceNormal_; }));
  // And finally the curves. Behind buttons, following the stabilizer's
  // precedent (StabilizerTask_log_gui.cpp:272): a plot costs bandwidth every
  // cycle it is open, and only a GUI client that implements plots shows one at
  // all, so it is opened on request rather than by default.
  auto plotButtons = [this](const char * title, const char * plotName, int axis)
  {
    return std::make_pair(
        Button(title,
               [this, plotName, axis]()
               {
                 gui()->addPlot(plotName, plot::X("t [s]", [this]() { return elapsed_; }),
                                plot::Y(
                                    "estimated [deg]",
                                    [this, axis]()
                                    { return mc_rtc::constants::toDeg(attitudeMonitorEstimatedRP_[axis]); },
                                    Color::Green),
                                plot::Y(
                                    "reference [deg]",
                                    [this, axis]()
                                    { return mc_rtc::constants::toDeg(attitudeMonitorReferenceRP_[axis]); },
                                    Color::Gray));
               }),
        Button(std::string{"Stop "} + plotName, [this, plotName]() { gui()->removePlot(plotName); }));
  };
  auto roll = plotButtons("Plot roll", "Roll", 0);
  auto pitch = plotButtons("Plot pitch", "Pitch", 1);
  gui()->addElement({"Rolling Contact", "Attitude"}, ElementsStacking::Horizontal, roll.first, roll.second);
  gui()->addElement({"Rolling Contact", "Attitude"}, ElementsStacking::Horizontal, pitch.first, pitch.second);
}

MCRollingContactController::~MCRollingContactController()
{
  // Restore terminal settings and join the keyboard thread before tearing down
  // any controller state that the interactive loop can indirectly observe.
  keyboard_.reset();
  // TVM keeps graph nodes that refer to tasks and constraint functions. Remove
  // every derived-class object while it is still alive, before C++ destroys
  // these members and eventually the base-class solver.
  solver().removeTask(basePositionTask_);
  solver().removeTask(baseOrientationTask_);
  solver().removeTask(postureTask);
  solver().removeConstraintSet(*rolling_);
  solver().removeConstraintSet(*dynamics_);
}

std::vector<mc_rbdyn::RollingContactDescription> MCRollingContactController::makeWheels() const
{
  std::vector<mc_rbdyn::RollingContactDescription> wheels;
  const std::vector<std::string> names = fourSteering_
                                             ? std::vector<std::string>{"front_left", "front_right", "rear_left",
                                                                        "rear_right"}
                                             : std::vector<std::string>{"left", "right"};
  const std::vector<std::string> surfaces = fourSteering_
                                                ? std::vector<std::string>{"FrontLeftWheel", "FrontRightWheel",
                                                                           "RearLeftWheel", "RearRightWheel"}
                                                : std::vector<std::string>{"LeftWheel", "RightWheel"};
  for(size_t i = 0; i < names.size(); ++i)
  {
    const auto & name = names[i];
    mc_rbdyn::RollingContactDescription wheel;
    wheel.name = name;
    wheel.carrierFrame = name + "_carrier";
    wheel.wheelBody = name + "_wheel";
    wheel.driveJoint = name + "_drive";
    if(fourSteering_) { wheel.steeringJoint = name + "_steer"; }
    const auto * surface = dynamic_cast<const mc_rbdyn::CylindricalSurface *>(&robot().surface(surfaces[i]));
    if(!surface)
    {
      mc_rtc::log::error_and_throw<std::invalid_argument>("RollingContact surface {} must be cylindrical", surfaces[i]);
    }
    wheel.radius = surface->radius();
    if(!fourSteering_ && scenario_ == "unequal_radii") { wheel.radius = name == "left" ? 0.18 : 0.22; }
    wheel.width = surface->width();
    wheel.friction = 0.8;
    wheel.validate();
    wheels.push_back(std::move(wheel));
  }
  return wheels;
}

void MCRollingContactController::reset(const ControllerResetData & data)
{
  MCController::reset(data);
  elapsed_ = 0.0;
  lastSolverSuccess_ = false;
  contactFallback_ = false;
  keyboardCaptureWasRunning_ = false;
  commandedTwist_.setZero();
  updateTimeMs_ = 0.0;
  solveAndBuildTimeMs_ = 0.0;
  totalTimeMs_ = 0.0;
  maxRollingResidual_ = 0.0;
  maxLateralResidual_ = 0.0;
  lateralSlackNorm_ = 0.0;
  minFrictionMargin_ = 0.0;
  referenceLinearSpeed_ = 0.0;
  referenceLateralSpeed_ = 0.0;
  referenceYawRate_ = 0.0;
  guiForwardCommand_ = 0.0;
  guiLateralCommand_ = 0.0;
  guiYawCommand_ = 0.0;
  if(keyboard_) { keyboard_->clear(); }
  dynamicsResidual_ = 0.0;
  floatingBaseEffortNorm_ = 0.0;
  basePositionTarget_ = robot().frame("chassis").position().translation();
  baseReferenceVelocity_.setZero();
  baseTrackingVelocity_.setZero();
  baseReferenceAngularVelocity_.setZero();
  // Start the accumulated yaw reference at the measured chassis heading. This
  // keeps the first Q/E command continuous even when the simulator is reset
  // with a non-zero initial orientation.
  const double initialYaw = measuredWorldYaw();
  baseYawTarget_ = std::isfinite(initialYaw) ? initialYaw : 0.0;
  keyboardYawTarget_ = baseYawTarget_;
  keyboardYawCorrection_ = 0.0;
  keyboardYawError_ = 0.0;
  diagnosticsValid_ = false;
  invalidReason_ = "not-run";
  steeringRateReferences_.assign(wheels_.size(), 0.0);
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    const auto joint = robot().jointIndexByName(wheels_[i].driveJoint);
    driveTargets_[i] = robot().mbc().q[joint][0];
    wheelReferenceRates_[i] = 0.0;
    if(!wheels_[i].steeringJoint.empty())
    {
      const auto steering = robot().jointIndexByName(wheels_[i].steeringJoint);
      steeringTargets_[i] = robot().mbc().q[steering][0];
    }
    modeManagers_[i].reset(wheels_[i].mode, wheels_[i].mode,
                           wheels_[i].mode == mc_rbdyn::RollingContactMode::Detached ? 0.0 : 1.0);
    appliedActivations_[i] = modeManagers_[i].state().activation;
    hardPromotionPending_[i] = wheels_[i].mode == mc_rbdyn::RollingContactMode::Detached;
    rolling_->mode(wheels_[i].name, wheels_[i].mode, appliedActivations_[i]);
    dynamics_->mode(wheels_[i].name, wheels_[i].mode);
  }
  std::fill(rollingResiduals_.begin(), rollingResiduals_.end(), 0.0);
  std::fill(lateralResiduals_.begin(), lateralResiduals_.end(), 0.0);
  std::fill(normalResiduals_.begin(), normalResiduals_.end(), 0.0);
  std::fill(accelerationResiduals_.begin(), accelerationResiduals_.end(), 0.0);
  std::fill(lateralAccelerationResiduals_.begin(), lateralAccelerationResiduals_.end(), 0.0);
  std::fill(normalAccelerationResiduals_.begin(), normalAccelerationResiduals_.end(), 0.0);
  std::fill(hardPromotionResiduals_.begin(), hardPromotionResiduals_.end(), 0.0);
  std::fill(normalForces_.begin(), normalForces_.end(), 0.0);
  std::fill(tangentialForces_.begin(), tangentialForces_.end(), 0.0);
  std::fill(frictionMargins_.begin(), frictionMargins_.end(), 0.0);
  std::fill(driveTorques_.begin(), driveTorques_.end(), 0.0);
  std::fill(driveTorqueMargins_.begin(), driveTorqueMargins_.end(), 0.0);
  for(auto & measurement : externalMeasurements_) { measurement = {}; }
  postureTask->reset();
  baseOrientationTask_->reset();
  basePositionTask_->reset();
}

void MCRollingContactController::setCommandedTwist(const Eigen::Vector3d & twist)
{
  if(!twist.allFinite())
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>("RollingContact commanded twist must be finite");
  }
  commandedTwist_ = twist;
}

double MCRollingContactController::measuredWorldYaw() const
{
  // E_0_b^T * e_x: see the doc comment on the declaration for the convention.
  const Eigen::Vector3d worldForward = robot().posW().rotation().transpose().col(0);
  return std::atan2(worldForward.dot(terrainTangentY_), worldForward.dot(terrainTangentX_));
}

void MCRollingContactController::saturateYawTargetAgainstMeasuredHeading()
{
  // Reference governor on the accumulated heading target.
  //
  // Every branch that reaches here integrates the *commanded* yaw rate, with no
  // feedback from the chassis. Whenever the chassis cannot deliver that rate -
  // and a four-steering chassis routinely cannot, because the hinges need
  // finite time and finite torque to reach the commanded instantaneous centre -
  // the orientation task's error grows at (commanded - measured) rad/s and is
  // unbounded. It does not merely become large: mc_tasks::OrientationTask's
  // error is sva::rotationError(), whose near-pi branch square-roots a quantity
  // that is only non-negative in exact arithmetic. Measured on the Ranger in
  // mc_mujoco, at |error| = pi - 0.0093 rad the intermediate is
  // s = [-1.11e-16, -1.11e-16, 1.0] and s.cwiseSqrt() returns NaN, the NaN
  // reaches the QP's linear term and the solve fails outright - with every
  // contact healthy, the chassis level, and 119 N of friction margin. That is
  // the failure this bound removes, and it removes it at the source: the QP is
  // never handed a reference whose error can reach the singular configuration.
  //
  // pi/2 is where an absolute heading reference stops carrying tracking
  // information (beyond a quarter turn the shortest-path error no longer says
  // which way the operator asked to go) and it keeps the rotation-error
  // Jacobian well conditioned: sinc_inv(pi/2) = 1.57 against 285 at the
  // pi - 0.011 rad where the NaN appears. A healthy run stays two orders of
  // magnitude below it - the worst heading error in a clean MuJoCo forward or
  // crab run is under 0.02 rad - so this is inert whenever tracking works, and
  // it deliberately does not hide the tracking loss: baseYawTarget_ resumes
  // integrating the instant the chassis catches up, and the shortfall stays
  // visible in RollingContact_base_orientation_eval.
  const double measuredYaw = measuredWorldYaw();
  if(!std::isfinite(measuredYaw) || !std::isfinite(baseYawTarget_)) { return; }
  constexpr double twoPi = 2.0 * 3.14159265358979323846;
  // measuredYaw is measuredWorldYaw(), +psi for a chassis yawed by psi (see
  // its doc comment). baseYawTarget_ is the world heading in every branch
  // that calls this, so the heading error is simply the difference.
  const double headingError = std::remainder(baseYawTarget_ - measuredYaw, twoPi);
  if(std::abs(headingError) <= maxYawTargetError_) { return; }
  baseYawTarget_ -= headingError - std::copysign(maxYawTargetError_, headingError);
}

/** Resolve this cycle's commanded planar twist [vx, vy, omega].
 *
 * Turns scenario_ into a twist - polling the keyboard when that is the
 * active scenario - republishes it through setCommandedTwist() on a
 * four-steering chassis, records it in the reference_* diagnostics, and
 * zeroes it while a contact fallback is latched.
 */
Eigen::Vector3d MCRollingContactController::resolveCommandedTwist()
{
  const double phase = 2.0 * 3.14159265358979323846 * elapsed_ / commandPeriod_;
  double forward = linearSpeed_;
  double lateral = 0.0;
  double yaw = yawRate_;
  if(scenario_ == "hold") { forward = yaw = 0.0; }
  else if(scenario_ == "reverse") { forward = -std::abs(linearSpeed_); yaw = 0.0; }
  else if(scenario_ == "forward" || scenario_ == "unequal_radii" || scenario_ == "infeasible_soft")
  {
    yaw = 0.0;
  }
  else if(scenario_ == "mode_cycle") { forward = yaw = 0.0; }
  else if(scenario_ == "turn_left") { forward = 0.0; yaw = std::abs(yawRate_); }
  else if(scenario_ == "turn_right") { forward = 0.0; yaw = -std::abs(yawRate_); }
  else if(scenario_ == "circle_left") { yaw = std::abs(yawRate_); }
  else if(scenario_ == "circle_right") { yaw = -std::abs(yawRate_); }
  else if(scenario_ == "sinusoid") { yaw *= std::sin(phase); }
  else if(scenario_ == "crab")
  {
    lateral = forward * std::sin(steeringAngle_);
    forward *= std::cos(steeringAngle_);
    yaw = 0.0;
  }
  else if(scenario_ == "steering_rate")
  {
    const double steering = steeringAngle_ * std::sin(phase);
    lateral = forward * std::sin(steering);
    forward *= std::cos(steering);
    yaw = 0.0;
  }
  else if(scenario_ == "incompatible") { yaw = 0.0; }
  else if(scenario_ == "ackermann_left") { yaw = std::abs(yawRate_); }
  else if(scenario_ == "ackermann_right") { yaw = -std::abs(yawRate_); }
  else if(scenario_ == "pure_yaw") { forward = 0.0; }
  else if(scenario_ == "keyboard")
  {
    const auto command = keyboard_->command();
    const bool keyboardRunning = keyboard_->running();
    // The exit key stops the capture worker. Clear GUI overrides at the same
    // poll so x is an unconditional stop, even when a ticker value is set.
    if(!keyboardRunning)
    {
      guiForwardCommand_ = 0.0;
      guiLateralCommand_ = 0.0;
      guiYawCommand_ = 0.0;
      if(keyboardCaptureWasRunning_)
      {
        // x is an explicit emergency stop. Retarget the held heading to the
        // measured pose at that instant so braking does not leave an
        // unreachable orientation error that would keep the wheels turning.
        const double measuredYaw = measuredWorldYaw();
        if(std::isfinite(measuredYaw)) { baseYawTarget_ = measuredYaw; }
      }
    }
    keyboardCaptureWasRunning_ = keyboardRunning;
    // RoboticsUtils uses right-positive lateral velocity and clockwise-positive
    // yaw. Convert to the robot convention: +Y left and +Z counter-clockwise.
    setCommandedTwist({command[0] + guiForwardCommand_, -command[1] + guiLateralCommand_,
                       -command[2] + guiYawCommand_});
  }
  else
  {
    // Unreachable as long as this chain covers every name the constructor
    // accepts, which it does today; kept so that adding a scenario to the
    // constructor's list and forgetting it here fails loudly instead of
    // silently driving at linearSpeed_/yawRate_. The condition used to be
    // scenario_ != "hold", which cannot be false here - "hold" is the first
    // branch of the chain.
    mc_rtc::log::error_and_throw<std::invalid_argument>("Unknown RollingContact scenario: {}", scenario_);
  }
  if(fourSteering_)
  {
    // The keyboard, the GUI and the scripted scenarios all reach the wheel
    // references through setCommandedTwist(). A scripted scenario republishes
    // its twist here every cycle. "keyboard" is excluded because it already
    // published its own twist above, from the key state plus the GUI offsets,
    // and "hold" because it publishes none at all: for those two, whatever the
    // operator (or a test harness) last commanded stays in effect.
    if(scenario_ != "hold" && scenario_ != "keyboard") { setCommandedTwist({forward, lateral, yaw}); }
    forward = commandedTwist_.x();
    lateral = commandedTwist_.y();
    yaw = commandedTwist_.z();
  }
  referenceLinearSpeed_ = forward;
  referenceLateralSpeed_ = lateral;
  referenceYawRate_ = yaw;
  if(contactFallback_)
  {
    forward = 0.0;
    lateral = 0.0;
    yaw = 0.0;
    referenceLinearSpeed_ = 0.0;
    referenceLateralSpeed_ = 0.0;
    referenceYawRate_ = 0.0;
    for(size_t i = 0; i < wheels_.size(); ++i)
    {
      const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
      driveTargets_[i] = robot().mbc().q[drive][0];
    }
  }
  return {forward, lateral, yaw};
}

/** Integrate the chassis pose targets and feed the two chassis tasks. */
void MCRollingContactController::updateChassisReference(const Eigen::Vector3d & twist)
{
  const double forward = twist.x();
  const double lateral = twist.y();
  const double yaw = twist.z();
  if(scenario_ == "keyboard")
  {
    if(closedLoopFeedback_)
    {
      // The Ranger wheel-drive convention is mirrored relative to the
      // controller's positive yaw convention. Integrating the requested yaw
      // here would therefore make the translational target turn opposite to
      // the measured chassis and the position error would grow without bound
      // during W+Q/W+E. In closed loop, the sensor is authoritative: follow
      // its current heading and let the orientation task's velocity reference
      // carry the operator's yaw command.
      // Deliberately -measuredWorldYaw(), not measuredWorldYaw(): baseYawTarget_
      // holds -psi in this branch only, mirrored back to +psi by
      // trajectoryHeadingYaw below. See the long comment there.
      const double measuredYaw = -measuredWorldYaw();
      if(std::isfinite(measuredYaw))
      {
        // Keep the logged target continuous while following the wrapped
        // sensor heading. The shortest delta is sufficient at the controller
        // rate and prevents a +/-pi crossing from reversing the trajectory
        // heading or failing the monotonic-yaw regression.
        baseYawTarget_ += std::remainder(measuredYaw - baseYawTarget_, 2.0 * 3.14159265358979323846);
      }
      else { baseYawTarget_ += yaw * solver().dt(); }
      // baseYawTarget_ above is re-snapped to the measured heading every
      // cycle (see the comment above), so it cannot also stand in for "the
      // heading the operator has asked for" - comparing it against the
      // measurement it was just copied from is always ~0. Accumulate that
      // separately, in measuredWorldYaw()'s +psi convention like yaw itself
      // (see the doc comment on keyboardYawTarget_), so
      // updateWheelReferences()'s pure-yaw feedback has a real target to
      // measure against.
      keyboardYawTarget_ =
          std::remainder(keyboardYawTarget_ + yaw * solver().dt(), 2.0 * 3.14159265358979323846);
    }
    else
    {
      // Keep the keyboard yaw reference unwrapped in open loop. The heading
      // below is periodic, but retaining the accumulated value avoids a
      // discontinuous scalar target at +/-pi.
      baseYawTarget_ += yaw * solver().dt();
      saturateYawTargetAgainstMeasuredHeading();
    }
  }
  else
  {
    baseYawTarget_ = std::remainder(baseYawTarget_ + yaw * solver().dt(), 2.0 * 3.14159265358979323846);
    saturateYawTargetAgainstMeasuredHeading();
  }
  // The two branches above leave baseYawTarget_ in two different conventions,
  // so the heading below has to undo the difference.
  //
  // Everywhere except closed-loop keyboard, baseYawTarget_ integrates the
  // commanded yaw rate, so it already is the world heading, measuredWorldYaw()'s
  // +psi convention (see its doc comment).
  //
  // Closed-loop keyboard instead copies -measuredWorldYaw(), i.e. -psi, the
  // negation of the world direction the chassis' +X axis actually travels, and
  // the sign has to be flipped back here. Without the flip a W+Q/W+E command
  // follows the mirrored circle and accumulates a metre-scale position-task
  // error even though the measured body-forward speed is correct.
  //
  // KeyboardClosedLoopYawTargetMirrorsTheMeasuredWorldHeading pins this.
  const double trajectoryHeadingYaw = scenario_ == "keyboard" && closedLoopFeedback_ ? -baseYawTarget_ : baseYawTarget_;
  const Eigen::Vector3d heading = std::cos(trajectoryHeadingYaw) * terrainTangentX_
                                  + std::sin(trajectoryHeadingYaw) * terrainTangentY_;
  const Eigen::Vector3d side = -std::sin(trajectoryHeadingYaw) * terrainTangentX_
                               + std::cos(trajectoryHeadingYaw) * terrainTangentY_;
  baseReferenceVelocity_ = forward * heading + lateral * side;
  baseReferenceAngularVelocity_ = yaw * terrainNormal_;
  const bool keyboardStoppedCommand = scenario_ == "keyboard" && std::abs(forward) < 1e-12
                                      && std::abs(lateral) < 1e-12 && std::abs(yaw) < 1e-12;
  if(keyboardStoppedCommand)
  {
    // A keyboard release means "hold here". The MuJoCo adapter may still
    // integrate the last interpolated wheel command for a few frames while
    // the transition grace period expires; refresh the hold target from the
    // measured chassis pose until the robot is fully stopped instead of
    // leaving a permanent position-task error that cannot be corrected with
    // zero wheel rates.
    basePositionTarget_ = robot().posW().translation();
  }
  basePositionTarget_ += solver().dt() * (forward * heading + lateral * side);
  // Feed the measured position error back into the chassis task. The
  // integrated target remains the user's requested trajectory, while this
  // correction compensates for the small velocity error introduced by the
  // simulator's wheel dynamics and makes the absolute target asymptotically
  // trackable instead of leaving a permanent position offset. It deliberately
  // does not reach the wheel references: those follow the commanded twist so a
  // constant command keeps a constant instantaneous centre of curvature.
  baseTrackingVelocity_ = baseReferenceVelocity_;
  if(closedLoopFeedback_ && scenario_ == "keyboard" && !contactFallback_ && positionFeedbackGain_ > 0.0)
  {
    const Eigen::Vector3d positionError = basePositionTarget_ - robot().posW().translation();
    const Eigen::Vector3d feedbackVelocity =
        positionFeedbackGain_ * (positionError.dot(heading) * heading + positionError.dot(side) * side);
    baseTrackingVelocity_ += feedbackVelocity;
  }
  basePositionTask_->position(basePositionTarget_);
  // The integrated position target provides the absolute trajectory, while
  // the velocity feed-forward makes the chassis follow that trajectory from
  // the first cycle. Without this, the target can move indefinitely while a
  // low-error position task continues to request zero chassis velocity.
  taskRefVel_ = baseTrackingVelocity_;
  basePositionTask_->refVel(taskRefVel_);
  Eigen::Matrix3d baseRotation;
  if(scenario_ == "keyboard" && closedLoopFeedback_)
  {
    // In closed-loop MuJoCo operation the measured chassis pose is the
    // authoritative state. Use it as the orientation task target and express
    // the keyboard yaw as a velocity reference; integrating an absolute
    // desired heading while the simulator lags would otherwise build an
    // unbounded orientation error and eventually make the contact QP fail.
    // orientation() below receives baseRotation.transpose(), so transpose the
    // measured world pose here to preserve the same body/world convention.
    // Passing the measured matrix without this transpose would command the
    // inverse yaw and fight the wheel-induced rotation.
    baseRotation = robot().posW().rotation().transpose();
  }
  else
  {
    baseRotation.col(0) = heading;
    baseRotation.col(1) = side;
    baseRotation.col(2) = terrainNormal_;
  }
  baseOrientationTask_->orientation(baseRotation.transpose());
  taskRefVel_ = baseReferenceAngularVelocity_;
  baseOrientationTask_->refVel(taskRefVel_);
}

/** Build this cycle's per-wheel drive and steering references. */
void MCRollingContactController::updateWheelReferences(const Eigen::Vector3d & twist)
{
  const double forward = twist.x();
  const double lateral = twist.y();
  const double yaw = twist.z();
  auto & targets = postureTargets_;
  // For a mixed translation+yaw command the steering angles encode the
  // requested instantaneous centre of curvature.  An absolute-heading
  // correction would continuously perturb that geometry whenever the pose
  // has a small phase offset, changing the radius and eventually exciting
  // rolling residuals.  Keep the correction for pure Q/E (where it removes a
  // heading bias) but leave the operator-selected radius untouched whenever
  // W/S/A/D is active.
  const bool keyboardPureYaw = std::abs(forward) < 1e-9 && std::abs(lateral) < 1e-9 && std::abs(yaw) > 1e-9;
  if(scenario_ == "keyboard" && closedLoopFeedback_ && keyboard_->running() && keyboardYawFeedbackGain_ > 0.0
     && keyboardPureYaw)
  {
    const double measuredYaw = measuredWorldYaw();
    if(std::isfinite(measuredYaw))
    {
      // keyboardYawTarget_, not baseYawTarget_: baseYawTarget_ is re-snapped to
      // this same measurement every cycle in updateChassisReference() (see the
      // comment there), so comparing it here was always ~0 and left this
      // feedback silently dead. keyboardYawTarget_ is the accumulator that
      // actually holds the operator's commanded heading. Use the wrapped
      // difference: the target is intentionally unwrapped for logging/
      // continuity, while the shortest local correction remains continuous
      // through every +/-pi crossing.
      const double yawError = std::remainder(keyboardYawTarget_ - measuredYaw, 2.0 * 3.14159265358979323846);
      keyboardYawError_ = yawError;
      keyboardYawCorrection_ = keyboardYawFeedbackGain_ * yawError;
    }
  }
  // The two branches below integrate the wheel targets that the generic
  // posture task follows as a position trajectory. For interactive operation
  // they additionally publish the desired wheel velocity as feed-forward: this
  // keeps the simulator tracking the command immediately instead of waiting
  // for a growing position error to generate acceleration through the posture
  // gain.
  const bool keyboardFeedForward = scenario_ == "keyboard" && solver().backend() == Backend::Tasks;
  if(keyboardFeedForward) { keyboardRefVel_.setZero(); }
  if(!fourSteering_)
  {
    const double halfTrack = 0.5 * std::abs(wheelOffsets_[0].y() - wheelOffsets_[1].y());
    std::array<double, 2> rates = {(forward - halfTrack * yaw) / wheels_[0].radius,
                                   (forward + halfTrack * yaw) / wheels_[1].radius};
    if(scenario_ == "infeasible_soft") { rates[0] += 4.0; }
    for(size_t i = 0; i < wheels_.size(); ++i)
    {
      driveTargets_[i] += rates[i] * solver().dt();
      wheelReferenceRates_[i] = rates[i];
      targets.at(wheels_[i].driveJoint)[0] = driveTargets_[i];
      if(keyboardFeedForward)
      {
        const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
        keyboardRefVel_(robot().mb().jointPosInDof(drive)) = rates[i];
      }
    }
  }
  else
  {
    // Every steering wheel reference is the exact inverse of the expanded
    // four-steering rolling rows for the commanded chassis twist. The QP then
    // tracks both rates through its soft rate rows, so there is no analytic
    // hinge IK, no branch-cut bookkeeping and no drive gating left here.
    const double dt = solver().dt();
    // The commanded twist plus the closed-loop keyboard yaw correction: the
    // wheel references are inverted from this, the chassis tasks from the
    // uncorrected command.
    const Eigen::Vector3d steeringTwist(forward, lateral, yaw + keyboardYawCorrection_);
    for(size_t i = 0; i < wheels_.size(); ++i)
    {
      mc_rbdyn::PlanarWheel planar;
      planar.offset = wheelOffsets_[i];
      planar.radius = wheels_[i].radius;
      planar.spinSign = wheels_[i].spinSign;

      const auto steeringJoint = robot().jointIndexByName(wheels_[i].steeringJoint);
      const double measuredSteering = robot().mbc().q[steeringJoint][0];
      planar.steeringAngle = measuredSteering;
      planar.steeringRate = robot().mbc().alpha[steeringJoint][0];

      // Wheel-centre velocity of the commanded planar twist, the same quantity
      // steeringWheelReference() inverts.
      const Eigen::Vector2d point(steeringTwist.x() - steeringTwist.z() * planar.offset.y(),
                                  steeringTwist.y() + steeringTwist.z() * planar.offset.x());
      auto reference = mc_rbdyn::steeringWheelReference(planar, steeringTwist, measuredSteering);

      // First-order convergence to the reference heading, saturated by the
      // hinge velocity limit. This is deltaDot^ref of the four-steering QP.
      const double steeringRate = std::clamp((reference.steeringAngle - measuredSteering) / steeringTimeConstant_,
                                             -maxSteeringRate_, maxSteeringRate_);

      // The rolling rate must stay consistent with the heading the wheel
      // actually has during the hinge slew, not with the heading it is
      // converging to: a rigid wheel can only roll along its current line, so
      // projecting the commanded wheel-centre velocity on the reference
      // heading would ask the QP to drive the chassis in a stale direction.
      // Both expressions coincide once the hinge has converged, which is what
      // makes the drive-gating of the previous open-loop layer unnecessary.
      double rollingRate =
          reference.commanded
              ? (std::cos(measuredSteering) * point.x() + std::sin(measuredSteering) * point.y())
                    / (wheels_[i].radius * wheels_[i].spinSign)
              : reference.rollingRate;
      if(scenario_ == "incompatible" && i == 0)
      {
        // Deliberately ask the first wheel for a rate no rigid rolling solution
        // can satisfy, so the soft longitudinal mode is exercised end to end.
        rollingRate += 4.0;
      }
      // Bound how fast the reference itself may change. A step in the commanded
      // twist otherwise asks the QP for the whole rate on the next cycle, which
      // saturates the contact friction cone for one tick.
      const double maxRateStep = driveAcceleration_ * dt;
      rollingRate = wheelReferenceRates_[i]
                    + std::clamp(rollingRate - wheelReferenceRates_[i], -maxRateStep, maxRateStep);

      steeringTargets_[i] = measuredSteering + steeringRate * dt;
      steeringRateReferences_[i] = steeringRate;
      wheelReferenceRates_[i] = rollingRate;
      driveTargets_[i] += rollingRate * dt;
      targets.at(wheels_[i].driveJoint)[0] = driveTargets_[i];
      targets.at(wheels_[i].steeringJoint)[0] = steeringTargets_[i];
      if(keyboardFeedForward)
      {
        const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
        keyboardRefVel_(robot().mb().jointPosInDof(drive)) = rollingRate;
        keyboardRefVel_(robot().mb().jointPosInDof(static_cast<int>(steeringJoint))) = steeringRate;
      }

      // The QP owns both rates from here on. It must receive the same rolling
      // rate that the posture target, the log and the simulator output carry:
      // reference.rollingRate is the pre-projection, pre-slew value.
      rolling_->rotatingRateReference(wheels_[i].name, rollingRate, steeringRate);
    }
  }
  postureTask->target(targets);
  if(keyboardFeedForward) { postureTask->refVel(keyboardRefVel_); }
}

void MCRollingContactController::updateReference()
{
  keyboardYawCorrection_ = 0.0;
  keyboardYawError_ = 0.0;
  const Eigen::Vector3d twist = resolveCommandedTwist();
  updateChassisReference(twist);
  updateWheelReferences(twist);
}

void MCRollingContactController::updateTerrainNormal()
{
  // Both constraints rebuild their contact geometry from the normal they hold
  // when update() runs, so a normal written once in the constructor and a
  // normal rewritten with the same value every cycle are indistinguishable to
  // the QP - which is the point. terrainNormal_ is still the configured
  // constant here; nothing estimates it yet. What this establishes is that the
  // per-cycle setter path itself is inert, so that when an estimate does start
  // flowing through it any change in behaviour is attributable to the estimate
  // and not to the plumbing.
  //
  // Cheap enough for the control loop: RollingContactConstraintOptions holds
  // only scalars and one fixed-size vector, so the copy-validate-assign in
  // RollingContactConstraint::terrainNormal() allocates nothing.
  rolling_->terrainNormal(terrainNormal_);
  dynamics_->terrainNormal(terrainNormal_);
}

Eigen::Vector3d MCRollingContactController::wheelOdometryTwist(const mc_rbdyn::Robot & measured, double & residual)
{
  const auto nrWheels = static_cast<Eigen::Index>(wheels_.size());
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    const auto drive = measured.jointIndexByName(wheels_[i].driveJoint);
    const double rate = measured.mbc().alpha[drive][0];
    // A differential chassis has no hinge; its wheel line is the chassis
    // forward axis, which is delta = 0 in exactly these rows.
    const double steering =
        wheels_[i].steeringJoint.empty()
            ? 0.0
            : measured.mbc().q[measured.jointIndexByName(wheels_[i].steeringJoint)][0];
    if(!std::isfinite(rate) || !std::isfinite(steering))
    {
      residual = std::numeric_limits<double>::quiet_NaN();
      return Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    }
    auto & planar = odometryWheels_[i];
    planar.offset = wheelOffsets_[i];
    planar.radius = wheels_[i].radius;
    planar.spinSign = wheels_[i].spinSign;
    planar.steeringAngle = steering;
    // The steering rate only enters steeringRollingMatrix()'s acceleration
    // bias, which this velocity-level solve does not use.
    planar.steeringRate = 0.0;
    odometryRates_[static_cast<Eigen::Index>(i)] = rate;
  }
  // Forwards, not inverted: rows.matrix * [vx, vy, omega, phidot...] = 0, so
  // the chassis twist solves A v = b with b carrying the measured wheel rates.
  const auto rows = mc_rbdyn::steeringRollingMatrix(odometryWheels_, Eigen::Vector3d::Zero());
  odometryMatrix_ = rows.matrix.leftCols<3>();
  odometryRhs_.noalias() = -rows.matrix.rightCols(nrWheels) * odometryRates_;
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    // sqrt(activation) on both sides is the weighted least squares the mode
    // manager's activation already means everywhere else in this controller:
    // the squared residual of each wheel enters the objective at `activation`.
    const double weight = std::sqrt(std::max(0.0, appliedActivations_[i]));
    const auto row = 2 * static_cast<Eigen::Index>(i);
    odometryMatrix_.middleRows<2>(row) *= weight;
    odometryRhs_.segment<2>(row) *= weight;
  }
  // Minimum-norm least squares. Rank-deficient inputs are ordinary here - one
  // attached wheel leaves the block rank 2, none at all leaves it zero - and
  // completeOrthogonalDecomposition() returns the minimum-norm solution for
  // both instead of throwing.
  const Eigen::Vector3d twist = odometryMatrix_.completeOrthogonalDecomposition().solve(odometryRhs_);
  residual = (odometryMatrix_ * twist - odometryRhs_).norm();
  return twist;
}

void MCRollingContactController::updateOdometry()
{
  // realRobot(), not robot(): the Encoder observer has refreshed it from this
  // tick's encoder packet by the time run() is reached, and reading it here
  // keeps the estimate on the measurement path in both the ticker and the
  // mc_mujoco deployment. robot().encoderVelocities() is deliberately not used:
  // the CTest lifecycle harness never populates that array (see the comment on
  // observerPipelineConfig() in test_controller_lifecycle.cpp).
  //
  // Computed before updateModes() so that the logged twist and the value the
  // observer pipeline read from the datastore earlier in the same tick are
  // built from the same activations.
  odometryTwist_ = wheelOdometryTwist(realRobot(), odometryResidual_);
}

void MCRollingContactController::updateAttitudeMonitor()
{
  // mc_rbdyn::rpyFromMat's roll and pitch read the third COLUMN of E_0_b and
  // nothing else:
  //     roll = atan2(E(1,2), E(2,2))     pitch = -asin(E(0,2))
  // Only yaw touches the first row. That column is the world vertical expressed
  // in body coordinates - which is exactly what a tilt estimator produces and
  // calls x2 - so applying these two expressions to the estimate and to the
  // reference compares like with like, with no yaw on either side and no Euler
  // ambiguity to resolve. A single lambda computes both for that reason.
  const auto rollPitch = [](const Eigen::Vector3d & tiltBody)
  {
    return Eigen::Vector2d{std::atan2(tiltBody.y(), tiltBody.z()),
                           -std::asin(std::clamp(tiltBody.x(), -1.0, 1.0))};
  };
  const Eigen::Matrix3d referenceRotation = realRobot().posW().rotation();
  attitudeMonitorReferenceRP_ = rollPitch(referenceRotation.col(2));
  // The chassis +z in world is E_0_b's third ROW, not its third column - the
  // column is the world vertical in body coordinates. Only the arrows use it.
  attitudeMonitorReferenceNormal_ = referenceRotation.row(2).transpose();
  // TiltBody, not Tilt: the observer has already carried x2 over to the IMU's
  // parent body with its own X_b_s, which this controller does not know. That
  // body is the Ranger's chassis, i.e. the floating base, so it shares
  // posW()'s frame and the two sides are directly comparable.
  const std::string tiltKey = "VelocityAidedTilt::TiltBody::" + robot().name();
  const std::string normalKey = "VelocityAidedTilt::Normal::" + robot().name();
  attitudeMonitorValid_ = datastore().has(tiltKey) && datastore().has(normalKey);
  if(!attitudeMonitorValid_)
  {
    // NaN, not zero and not a copy of the reference: a pipeline without a tilt
    // observer has no estimate, and both alternatives would draw a curve that
    // looks like a converged one.
    attitudeMonitorEstimatedRP_.setConstant(std::numeric_limits<double>::quiet_NaN());
    attitudeMonitorEstimatedNormal_ = attitudeMonitorReferenceNormal_;
    attitudeMonitorError_ = std::numeric_limits<double>::quiet_NaN();
    return;
  }
  const Eigen::Vector3d estimatedTilt = datastore().get<Eigen::Vector3d>(tiltKey);
  attitudeMonitorEstimatedRP_ = rollPitch(estimatedTilt);
  attitudeMonitorEstimatedNormal_ = datastore().get<Eigen::Vector3d>(normalKey);
  // The angle between the two tilts, which is the total attitude error the
  // roll/pitch pair splits: both vectors are unit, so their dot product is its
  // cosine. Frame-independent, and the quantity the ramp criteria are written
  // against.
  attitudeMonitorError_ = std::acos(std::clamp(estimatedTilt.dot(referenceRotation.col(2)), -1.0, 1.0));
}

void MCRollingContactController::updateModes()
{
  const bool wasContactFallback = contactFallback_;
  const bool keyboardCaptureActive = scenario_ == "keyboard" && keyboard_ && keyboard_->running();
  contactFallback_ = false;
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    externalMeasurements_[i].age += solver().dt();
    auto requested = mc_rbdyn::RollingContactMode::Rolling;
    if(scenario_ == "mode_cycle")
    {
      const double cycle = std::fmod(elapsed_, commandPeriod_) / commandPeriod_;
      if(cycle < 0.25) { requested = mc_rbdyn::RollingContactMode::Rolling; }
      else if(cycle < 0.5) { requested = mc_rbdyn::RollingContactMode::Fixed; }
      else if(cycle < 0.75) { requested = mc_rbdyn::RollingContactMode::Sliding; }
      else { requested = mc_rbdyn::RollingContactMode::Detached; }
    }
    modeManagers_[i].requestedMode(requested);
    if(lastSolverSuccess_)
    {
      mc_rbdyn::RollingContactModeObservation observation;
      observation.valid = diagnosticsValid_;
      observation.slipSpeed = std::hypot(rollingResiduals_[i], lateralResiduals_[i]);
      observation.rollingResidual = accelerationResiduals_[i];
      observation.normalForce = normalForces_[i];
      observation.frictionMargin = frictionMargins_[i];
      observation.torqueMargin = driveTorqueMargins_[i];
      const auto & measured = externalMeasurements_[i];
      if(measured.valid && measured.age <= 2.0 * solver().dt())
      {
        observation.slipSpeed = std::hypot(measured.rollingSlip, measured.lateralSlip);
        // A physics adapter supplies the authoritative contact velocity. The
        // inactive rolling-row acceleration residual can be intentionally
        // large in sliding/detached modes and must not veto physical recovery.
        // Hard-row promotion is gated separately below using that residual.
        observation.rollingResidual = measured.rollingSlip;
        observation.normalForce = measured.normalForce;
        observation.frictionMargin = wheels_[i].friction * measured.normalForce - measured.tangentialForce;
      }
      const bool hasFreshExternalMeasurement = measured.valid && measured.age <= 2.0 * solver().dt();
      if(requested != mc_rbdyn::RollingContactMode::Detached && !hasFreshExternalMeasurement)
      {
        // Neither the headless ticker nor mc_mujoco's default adapter has a
        // terrain contact sensor, so without a fresh external measurement the
        // only "normal force" available is normalForces_[i] - the QP's own
        // contact multiplier. That is a decision, not an observation, and it
        // must not be fed back as one:
        //
        // four coplanar wheel contacts leave the normal-force distribution with
        // a one-dimensional null space, the diagonal mode (+1, -1, -1, +1),
        // which produces no net force and no net moment and which nothing in
        // this QP's objective penalises (the lambda block carries only the
        // Tasks library's unconditional 1e-4 Hessian floor, eleven orders below
        // the rate rows at 4e7). The solver may therefore return zero on one
        // wheel while the chassis is perfectly level and every wheel is loaded.
        // Measured on the Ranger in mc_mujoco, a crab command drives the split
        // from 160/160/215/215 N to 183/0/205/341 N within four cycles - the
        // sum stays at the 736 N vehicle weight throughout, and the chassis
        // roll and pitch never leave 1e-5 rad.
        //
        // Reading that zero as a detachment closes a loop with no physical
        // content, and it is self-confirming: a Detached wheel's multiplier is
        // identically zero, so it can never re-enter, and the recovery probe
        // below is vetoed for as long as the rim turns faster than
        // recoverySpeed_. front_right then stays detached for the whole run.
        //
        // Use the same deterministic terrain-geometry probe the Detached ->
        // Rolling direction already used, in both directions. Only the normal
        // force is substituted: it is the one observation whose value the null
        // space moves freely and the only one that can force the irreversible
        // Detached verdict (desiredMode() returns Detached the instant
        // normalForce drops below normalForceExit, with no dwell). The friction
        // and torque margins are left alone for an attached wheel, so
        // Rolling -> Sliding still reacts to a QP solution that reaches its own
        // friction cone or its actuator bound, and so does the slip speed, which
        // is computed from the measured state and not from lambda at all. A
        // physics adapter or a force sensor that calls
        // RollingContact::SetMeasuredContact keeps full detection of all four.
        const auto & thresholds = modeManagers_[i].thresholds();
        observation.normalForce = thresholds.normalForceEnter + 1.0;
        if(modeManagers_[i].state().estimated == mc_rbdyn::RollingContactMode::Detached)
        {
          // Re-attachment additionally needs the two margins: desiredMode()'s
          // "recovered" test requires all of them at once, so leaving either at
          // its lambda value would deadlock a wheel whose contact rows are
          // switched off and whose multipliers are therefore identically zero.
          observation.frictionMargin = thresholds.frictionMarginEnter + 1.0;
          observation.torqueMargin = thresholds.torqueMarginEnter + 1.0;
        }
      }
      if(modeManagers_[i].state().estimated == mc_rbdyn::RollingContactMode::Detached
         && requested != mc_rbdyn::RollingContactMode::Detached && !keyboardCaptureActive)
      {
        const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
        if(std::abs(robot().mbc().alpha[drive][0]) * wheels_[i].radius > recoverySpeed_)
        {
          // A geometrically re-established contact is not yet safe to load
          // while the rim is moving quickly. Keep its QP force at zero until
          // the bounded safe-stop target has reduced wheel speed.
          observation.normalForce = 0.0;
        }
      }
      modeManagers_[i].update(observation, solver().dt());
    }
    const auto & state = modeManagers_[i].state();
    if(state.estimated == mc_rbdyn::RollingContactMode::Detached) { hardPromotionPending_[i] = true; }
    double appliedActivation = state.activation;
    if(hardPromotionPending_[i] && state.estimated == mc_rbdyn::RollingContactMode::Rolling
       && appliedActivation >= 1.0 - 1e-12)
    {
      // An activation below one keeps all would-be hard rows as a weighted
      // transition objective. Promote them only after the previous solution
      // demonstrates acceleration-level feasibility.
      appliedActivation = 1.0 - 1e-6;
    }
    appliedActivations_[i] = appliedActivation;
    wheels_[i].mode = state.estimated;
    wheels_[i].activation = state.activation;
    rolling_->mode(wheels_[i].name, state.estimated, appliedActivation);
    dynamics_->mode(wheels_[i].name, state.estimated);
    // A commanded twist change can change the instantaneous turning radius
    // while a steering hinge is still slewing.  The resulting short-lived
    // slip is a valid transition for the mode manager, not a reason to zero
    // the command: doing that here used to latch contactFallback_ and leave
    // keyboard Q/E+W/S combinations permanently stuck even though the QP
    // kept solving successfully.  Detached contacts and pending hard
    // promotion remain fail-safe; a transient Sliding/Detached estimate is
    // allowed to recover while the hinge that caused it is still moving.
    //
    // The transient is a property of the hinge motion, not of who issued the
    // twist: a scripted command that swings the reference steering angle
    // (e.g. crab to ackermann in one step) produces exactly the same
    // short-lived slip a keyboard operator triggers, so the recovery
    // allowance is keyed on the hinge still slewing rather than on
    // scenario_ == "keyboard". steeringRateReferences_[i] is only written by
    // the four-steering wheel loop of updateReference(), and the value read
    // here is one cycle old because run() calls updateModes() before
    // updateReference().
    const bool hingeSlewing = fourSteering_ && std::abs(steeringRateReferences_[i]) > 1e-6;
    const bool keyboardContactRecovery = (keyboardCaptureActive || hingeSlewing)
                                         && (state.estimated == mc_rbdyn::RollingContactMode::Detached
                                             || state.estimated == mc_rbdyn::RollingContactMode::Sliding);
    contactFallback_ = contactFallback_
                       || (requested == mc_rbdyn::RollingContactMode::Rolling
                           && (state.estimated == mc_rbdyn::RollingContactMode::Detached
                               || hardPromotionPending_[i])
                           && !keyboardContactRecovery);
  }
  bool anyPromotionPending = false;
  bool allReadyForHardPromotion = diagnosticsValid_;
  const double promotionResidualLimit =
      rolling_->options().longitudinal == mc_solver::RollingContactLongitudinal::Hard ? recoveryResidual_
                                                                                     : recoveryTransverseResidual_;
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    if(!hardPromotionPending_[i]) { continue; }
    anyPromotionPending = true;
    const auto & state = modeManagers_[i].state();
    const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
    allReadyForHardPromotion =
        allReadyForHardPromotion && state.estimated == mc_rbdyn::RollingContactMode::Rolling
        && state.activation >= 1.0 - 1e-12 && hardPromotionResiduals_[i] <= promotionResidualLimit
        && std::abs(robot().mbc().alpha[drive][0]) * wheels_[i].radius <= recoverySpeed_;
  }
  if(anyPromotionPending && allReadyForHardPromotion)
  {
    std::fill(hardPromotionPending_.begin(), hardPromotionPending_.end(), false);
  }
  if(contactFallback_ && !wasContactFallback)
  {
    const auto & chassis = robot().frame("chassis").position();
    basePositionTarget_ = chassis.translation();
    // "chassis" is the floating-base frame for every rolling-contact robot
    // (RollingContactControllerSynchronizesFloatingBase pins
    // frame("chassis").position() == posW()), so measuredWorldYaw() reads the
    // same pose. Leave baseYawTarget_ unchanged in the degenerate NaN case
    // rather than snapping it to an arbitrary heading mid-recovery.
    const double worldYaw = measuredWorldYaw();
    baseYawTarget_ = std::isfinite(worldYaw) ? worldYaw : baseYawTarget_;
  }
  if(contactFallback_ != wasContactFallback)
  {
    // Transitional rows remain soft until their previous-cycle residuals are
    // safe to promote. Give those rows enough authority to demonstrate
    // feasibility against ordinary tracking tasks on both QP backends.
    rolling_->rollingWeight(contactFallback_ ? recoveryRollingWeight_ : rollingWeight_);
  }
}

void MCRollingContactController::safeStop(const std::string & reason)
{
  diagnosticsValid_ = false;
  invalidReason_ = reason;
  lastSolverSuccess_ = false;
  referenceLinearSpeed_ = 0.0;
  referenceLateralSpeed_ = 0.0;
  referenceYawRate_ = 0.0;
  guiForwardCommand_ = 0.0;
  guiLateralCommand_ = 0.0;
  guiYawCommand_ = 0.0;
  commandedTwist_.setZero();
  if(keyboard_) { keyboard_->clear(); }
  basePositionTarget_ = robot().posW().translation();
  baseReferenceVelocity_.setZero();
  baseTrackingVelocity_.setZero();
  baseReferenceAngularVelocity_.setZero();
  basePositionTask_->position(basePositionTarget_);
  taskRefVel_.setZero();
  basePositionTask_->refVel(taskRefVel_);
  baseOrientationTask_->orientation(robot().posW().rotation());
  baseOrientationTask_->refVel(taskRefVel_);
  std::fill(wheelReferenceRates_.begin(), wheelReferenceRates_.end(), 0.0);
  auto & targets = postureTargets_;
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
    driveTargets_[i] = robot().mbc().q[drive][0];
    targets.at(wheels_[i].driveJoint)[0] = driveTargets_[i];
    if(!wheels_[i].steeringJoint.empty())
    {
      const auto steering = robot().jointIndexByName(wheels_[i].steeringJoint);
      steeringTargets_[i] = robot().mbc().q[steering][0];
      steeringRateReferences_[i] = 0.0;
      targets.at(wheels_[i].steeringJoint)[0] = robot().mbc().q[steering][0];
    }
    rolling_->rotatingRateReference(wheels_[i].name, 0.0, 0.0);
    modeManagers_[i].reset(mc_rbdyn::RollingContactMode::Detached, mc_rbdyn::RollingContactMode::Detached, 0.0);
    hardPromotionPending_[i] = true;
    appliedActivations_[i] = 0.0;
    rolling_->mode(wheels_[i].name, mc_rbdyn::RollingContactMode::Detached);
    dynamics_->mode(wheels_[i].name, mc_rbdyn::RollingContactMode::Detached);
    wheels_[i].mode = mc_rbdyn::RollingContactMode::Detached;
    wheels_[i].activation = 0.0;
  }
  std::fill(normalForces_.begin(), normalForces_.end(), 0.0);
  std::fill(tangentialForces_.begin(), tangentialForces_.end(), 0.0);
  std::fill(frictionMargins_.begin(), frictionMargins_.end(), 0.0);
  std::fill(driveTorques_.begin(), driveTorques_.end(), 0.0);
  std::fill(driveTorqueMargins_.begin(), driveTorqueMargins_.end(), 0.0);
  postureTask->target(targets);
}

void MCRollingContactController::updateDiagnostics(bool solverSuccess)
{
  lastSolverSuccess_ = solverSuccess;
  diagnosticsValid_ = solverSuccess;
  invalidReason_ = solverSuccess ? std::string{} : std::string{"solver-failure"};
  maxRollingResidual_ = 0.0;
  maxLateralResidual_ = 0.0;
  lateralSlackNorm_ = 0.0;
  minFrictionMargin_ = std::numeric_limits<double>::infinity();
  const auto & geometry = rolling_->geometryResults();
  // Pre-sized in the constructor: rbd::dofToVector() would allocate a fresh
  // nrDof vector every cycle, and it is exactly this buffer filled in place.
  Eigen::VectorXd & alphaD = alphaDBuffer_;
  if(solver().backend() == Backend::TVM)
  {
    alphaD = robot().tvmRobot().alphaD()->value();
    dynamics_->dynamicFunction().updateValue();
    dynamicsResidual_ = dynamics_->dynamicFunction().value().lpNorm<Eigen::Infinity>();
    const auto & effort = robot().tvmRobot().tau()->value();
    const Eigen::Index floatingDof = robot().mb().joint(0).type() == rbd::Joint::Free ? 6 : 0;
    floatingBaseEffortNorm_ = floatingDof == 0 ? 0.0 : effort.head(floatingDof).norm();
  }
  else { rbd::paramToVector(robot().mbc().alphaD, alphaD); }
  lateralSlackNorm_ = rolling_->lateralSlack(alphaD).norm();
  for(size_t i = 0; i < geometry.size(); ++i)
  {
    rollingResiduals_[i] = geometry[i].rollingDirection.dot(geometry[i].slipVelocity);
    lateralResiduals_[i] = geometry[i].lateralDirection.dot(geometry[i].slipVelocity);
    normalResiduals_[i] = geometry[i].normalDirection.dot(geometry[i].velocityResidual);
    const Eigen::Vector3d accelerationResidual = geometry[i].rollingMatrix * alphaD - geometry[i].rhs;
    accelerationResiduals_[i] = std::abs(accelerationResidual.x());
    lateralAccelerationResiduals_[i] = std::abs(accelerationResidual.y());
    normalAccelerationResiduals_[i] = std::abs(accelerationResidual.z());
    // Normal reattachment is certified by the fresh simulator/contact-sensor
    // normal-force hysteresis. A soft normal acceleration row can legitimately
    // retain the gravity term while the contact is unloaded, so using it as a
    // hard-promotion gate would deadlock recovery. Lateral compatibility and,
    // for hard rolling, longitudinal compatibility are checked algebraically.
    hardPromotionResiduals_[i] = lateralAccelerationResiduals_[i];
    if(rolling_->options().longitudinal == mc_solver::RollingContactLongitudinal::Hard)
    {
      hardPromotionResiduals_[i] = std::max(hardPromotionResiduals_[i], accelerationResiduals_[i]);
    }
    maxRollingResidual_ = std::max(maxRollingResidual_, std::abs(rollingResiduals_[i]));
    maxLateralResidual_ = std::max(maxLateralResidual_, std::abs(lateralResiduals_[i]));
    normalForces_[i] = 0.0;
    tangentialForces_[i] = 0.0;
    frictionMargins_[i] = 0.0;
    const auto joint = robot().jointIndexByName(wheels_[i].driveJoint);
    driveTorques_[i] = robot().jointTorque()[joint][0];
    driveTorqueMargins_[i] = std::min(robot().tu()[joint][0] - driveTorques_[i],
                                      driveTorques_[i] - robot().tl()[joint][0]);
  }
  if(solverSuccess)
  {
    if(solver().backend() == Backend::Tasks)
    {
      const auto & tasksSolver = static_cast<mc_solver::TasksQPSolver &>(solver());
      const auto & lambda = tasksSolver.solver().lambdaVec();
      for(size_t i = 0; i < wheels_.size(); ++i)
      {
        const auto & wheel = wheels_[i];
        const int begin = dynamics_->lambdaBegin(wheel.name) - tasksSolver.data().lambdaBegin();
        if(begin >= 0 && begin + dynamics_->lambdaCount(wheel.name) <= lambda.size())
        {
          normalForces_[i] = dynamics_->normalForce(wheel.name, lambda.segment(begin, 8));
          tangentialForces_[i] = dynamics_->tangentialForce(wheel.name, lambda.segment(begin, 8));
          frictionMargins_[i] = dynamics_->frictionMargin(wheel.name, lambda.segment(begin, 8));
          minFrictionMargin_ = std::min(minFrictionMargin_, frictionMargins_[i]);
        }
      }
    }
    else
    {
      for(size_t i = 0; i < wheels_.size(); ++i)
      {
        normalForces_[i] = dynamics_->normalForce(wheels_[i].name);
        tangentialForces_[i] = dynamics_->tangentialForce(wheels_[i].name);
        frictionMargins_[i] = dynamics_->frictionMargin(wheels_[i].name);
        minFrictionMargin_ = std::min(minFrictionMargin_, frictionMargins_[i]);
      }
    }
  }
  if(!std::isfinite(minFrictionMargin_)) { minFrictionMargin_ = 0.0; }
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    diagnosticsValid_ = diagnosticsValid_ && std::isfinite(rollingResiduals_[i])
                        && std::isfinite(lateralResiduals_[i]) && std::isfinite(normalResiduals_[i])
                        && std::isfinite(accelerationResiduals_[i])
                        && std::isfinite(lateralAccelerationResiduals_[i])
                        && std::isfinite(normalAccelerationResiduals_[i]) && std::isfinite(hardPromotionResiduals_[i])
                        && std::isfinite(normalForces_[i])
                        && std::isfinite(tangentialForces_[i]) && std::isfinite(frictionMargins_[i])
                        && std::isfinite(driveTorques_[i]) && std::isfinite(driveTorqueMargins_[i]);
  }
  if(!diagnosticsValid_ && invalidReason_.empty()) { invalidReason_ = "non-finite-diagnostic"; }
}

void MCRollingContactController::syncControlRobotFromSensors()
{
  // mc_mujoco publishes joint encoders and the FloatingBase body sensor on the
  // control robot only. Feed them into the control MBC so the QP built this
  // tick sees the measured state (closed-loop feedback), matching how a real
  // hardware interface's encoder/IMU packet would be consumed. This is
  // independent from state observation: realRobot() is estimated by the
  // Encoder/BodySensor observer pipeline (see ObserverPipelines in the
  // controller configuration), which runs before MCController::run() and
  // reads these same raw sensor values off the control robot, so it does not
  // need (and must not be overwritten by) a copy from here.
  auto & measuredRobot = robot();
  const auto & encoders = measuredRobot.encoderValues();
  const auto & encoderVelocities = measuredRobot.encoderVelocities();
  const auto & refJointOrder = measuredRobot.refJointOrder();
  for(size_t refIndex = 0; refIndex < refJointOrder.size(); ++refIndex)
  {
    const auto & jointName = refJointOrder[refIndex];
    if(!measuredRobot.hasJoint(jointName) || refIndex >= encoders.size()) { continue; }
    const auto jointIndex = measuredRobot.jointIndexByName(jointName);
    if(measuredRobot.mb().joint(jointIndex).dof() != 1) { continue; }
    measuredRobot.mbc().q[jointIndex][0] = encoders[refIndex];
    if(refIndex < encoderVelocities.size()) { measuredRobot.mbc().alpha[jointIndex][0] = encoderVelocities[refIndex]; }
  }

  // The floating-base estimate is independent from the articulated encoder
  // packet. A simulator (or a hardware interface during startup) can publish
  // an IMU pose/velocity before every wheel encoder is valid; do not keep the
  // base at its old pose in that case. Joint encoders are validated separately
  // in the loop above.
  if(measuredRobot.mb().joint(0).type() == rbd::Joint::Type::Free && measuredRobot.hasBodySensor("FloatingBase"))
  {
    const auto & sensor = measuredRobot.bodySensor("FloatingBase");
    const auto & position = sensor.position();
    const auto & orientation = sensor.orientation();
    if(position.allFinite() && orientation.coeffs().allFinite() && orientation.norm() > 1e-12)
    {
      // BodySensor orientations are inertial-to-sensor. Robot::posW takes
      // that same world-to-base transform and internally writes the inverse
      // quaternion, matching MuJoCo's free-joint qpos convention.
      Eigen::Quaterniond worldToBody = orientation;
      worldToBody.normalize();
      measuredRobot.posW(sva::PTransformd(worldToBody, position));
      const auto & angularVelocity = sensor.angularVelocity();
      const auto & linearVelocity = sensor.linearVelocity();
      if(angularVelocity.allFinite() && linearVelocity.allFinite())
      {
        // BodySensor readings are expressed in the inertial frame, whereas
        // RBDyn's free-joint alpha stores angular/linear components in the
        // body frame. Passing the world linear velocity through unchanged
        // works while the chassis faces +X, but it appears as a growing
        // lateral rolling error as the robot turns and makes the mode manager
        // promote otherwise healthy wheels to Sliding/Detached.
        const Eigen::Matrix3d worldToBodyRotation = worldToBody.toRotationMatrix();
        const Eigen::Vector3d bodyAngularVelocity = worldToBodyRotation * angularVelocity;
        const Eigen::Vector3d bodyLinearVelocity = worldToBodyRotation * linearVelocity;
        measuredRobot.mbc().alpha[0] = {bodyAngularVelocity.x(), bodyAngularVelocity.y(), bodyAngularVelocity.z(),
                                        bodyLinearVelocity.x(),  bodyLinearVelocity.y(),  bodyLinearVelocity.z()};
      }
    }
  }
  measuredRobot.forwardKinematics();
  measuredRobot.forwardVelocity();
}

bool MCRollingContactController::run()
{
  const auto begin = std::chrono::steady_clock::now();
  bool success = false;
  try
  {
    if(closedLoopFeedback_)
    {
      syncControlRobotFromSensors();
    }
    for(size_t i = 0; i < wheels_.size(); ++i)
    {
      const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
      measuredDrivePositions_[i] = robot().mbc().q[drive][0];
    }
    updateOdometry();
    updateAttitudeMonitor();
    updateTerrainNormal();
    updateModes();
    updateReference();
    success = MCController::run();
    if(success && scenario_ == "keyboard")
    {
      // mc_mujoco drives articulated joints from the controller's q/alpha
      // output. Keep the wheel position output at the measured position and
      // expose the keyboard velocity as alpha; otherwise the QP's alphaD
      // (which is only an acceleration result) is integrated from zero and
      // the simulator never receives the requested wheel speed.
      for(size_t i = 0; i < wheels_.size(); ++i)
      {
        const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
        robot().mbc().q[drive][0] = measuredDrivePositions_[i];
        robot().mbc().alpha[drive][0] = wheelReferenceRates_[i];
        if(!wheels_[i].steeringJoint.empty())
        {
          const auto steering = robot().jointIndexByName(wheels_[i].steeringJoint);
          robot().mbc().q[steering][0] = steeringTargets_[i];
          // mc_mujoco consumes alpha as the actuator velocity reference. A
          // position-only target is not sufficient for this joint (the
          // simulator would keep the steering angle at its old value), so
          // publish the same bounded rate reference the QP was given.
          robot().mbc().alpha[steering][0] = steeringRateReferences_[i];
        }
      }
      // Keep the free-joint q/alpha state measured. mc_mujoco only consumes
      // the actuated wheel entries from this MBC; it does not consume a
      // floating-base command. Writing the world-frame task references into
      // alpha[0] here would both corrupt the measured state (RBDyn stores the
      // free-joint velocity in body coordinates) and make a subsequent Q/E to
      // A/W transition appear as a large spurious chassis velocity.
      robot().forwardKinematics();
      robot().forwardVelocity();
    }
    if(success && closedLoopFeedback_ && robot().mb().joint(0).type() == rbd::Joint::Type::Free)
    {
      // The free joint is measured state, not an actuator command, in the
      // mc_mujoco adapter. MCController::run() integrates a one-step task
      // prediction into the control MBC; restore the measured root after
      // producing the wheel commands so the next cycle and ff_q never use
      // that prediction as if it were an observed chassis motion. realRobot()
      // is the observer pipeline's estimate for this tick: the Encoder and
      // BodySensor observers run (via runObserverPipelines()) before this
      // run() is called, so realRobot().mbc().q[0]/alpha[0] already reflect
      // the same sensor packet syncControlRobotFromSensors() just wrote into
      // robot(). A caller that drives this controller without running the
      // configured ObserverPipelines every cycle will restore a stale free
      // joint here.
      robot().mbc().q[0] = realRobot().mbc().q[0];
      robot().mbc().alpha[0] = realRobot().mbc().alpha[0];
      robot().forwardKinematics();
      robot().forwardVelocity();
    }
    updateDiagnostics(success);
    if(!success) { safeStop("solver-failure"); }
  }
  catch(const std::exception & error)
  {
    safeStop(std::string{"runtime-exception: "} + error.what());
    mc_rtc::log::error("RollingContact controller contained runtime failure: {}", error.what());
  }
  elapsed_ += solver().dt();
  totalTimeMs_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
  solveAndBuildTimeMs_ = solver().solveAndBuildTime();
  updateTimeMs_ = std::max(0.0, totalTimeMs_ - solveAndBuildTimeMs_);
  return success;
}

} // namespace mc_control

MULTI_CONTROLLERS_CONSTRUCTOR(
    "RollingContact",
    mc_control::MCRollingContactController(rm, dt, config, mc_control::MCController::Backend::Tasks),
    "RollingContact_TVM",
    mc_control::MCRollingContactController(rm, dt, config, mc_control::MCController::Backend::TVM))
