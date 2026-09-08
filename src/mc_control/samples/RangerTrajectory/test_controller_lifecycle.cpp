/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

/** Lifecycle checks for the RangerTrajectory FSM sample.
 *
 * The sample has no C++ behaviour of its own, so everything asserted here is a
 * property of etc/RangerTrajectory.in.yaml driven through a real
 * mc_control::fsm::Controller: the state sequence, the geometry the generated
 * control points actually produce, the invariants the dimWeight choice is
 * supposed to hold, and -- honestly labelled as such -- the two reasons the
 * chassis does not follow the curved states. See the KNOWN LIMITATION block at
 * the top of the sample's YAML.
 */

#include "mc_ranger_trajectory_controller.h"

#include <mc_observers/ObserverLoader.h>
#include <mc_rbdyn/RobotLoader.h>
#include <mc_solver/RollingContactDynamicsConstraint.h>
#include <mc_solver/TasksQPSolver.h>
#include <mc_tasks/BSplineTrajectoryTask.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{

constexpr double dt = 0.005;
constexpr double circleDuration = 14.0;
constexpr double lineDuration = 4.5;
constexpr double lissajousDuration = 34.0;
constexpr double circleRadius = 1.0;
constexpr double lineLength = 2.0;
constexpr double lissajousA = 1.0;
constexpr double lissajousB = 1.0;
/** URDF effort limit of the four *_drive joints. */
constexpr double driveTorqueLimit = 35.0;

const std::vector<std::string> wheelNames = {"front_left", "front_right", "rear_left", "rear_right"};

/** Opens up the two protected members the assertions need.
 *
 * The FSM executor carries the current state name and the loaded constraint
 * sets carry the rolling-contact dynamics object whose contact normal forces
 * tell whether a wheel came off the ground. Neither is needed by the sample
 * itself, so they stay protected in mc_rtc and are exposed here rather than by
 * widening a shared interface for a test.
 */
struct ProbeController : public RangerTrajectoryController
{
  using mc_control::fsm::Controller::executor_;
  using mc_control::MCController::constraints_;
  using RangerTrajectoryController::RangerTrajectoryController;
};

mc_rbdyn::RobotModulePtr robotModule()
{
  static const bool configured = []()
  {
    mc_rbdyn::RobotLoader::clear();
    mc_rbdyn::RobotLoader::update_robot_module_path({RANGER_TRAJECTORY_ROBOT_MODULE_PATH});
    return true;
  }();
  (void)configured;
  return mc_rbdyn::RobotLoader::get_robot_module("RollingContactRangerMiniV3");
}

void loadObserverModules()
{
  static const bool configured = []()
  {
    mc_observers::ObserverLoader::clear();
    mc_observers::ObserverLoader::update_module_path({RANGER_TRAJECTORY_OBSERVER_MODULE_PATH});
    return true;
  }();
  (void)configured;
}

/** The sample's own configuration, with the observer pipeline switched to
 * "control" mode.
 *
 * The shipped pipeline reads "encoderValues"/"encoderVelocities" because a
 * simulator publishes a real encoder packet every tick. This harness
 * constructs the controller directly, so nothing fills those arrays and
 * EncoderObserver::run() throws on an empty one
 * (mc_observers/EncoderObserver.cpp:88-97). "control" mirrors mbc().q/alpha
 * instead, which keeps Encoder genuinely running ahead of BodySensor in the
 * pipeline. Everything else -- the constraints, the states, the transitions --
 * is the file the sample installs.
 */
mc_rtc::Configuration sampleConfiguration()
{
  mc_rtc::Configuration config(RANGER_TRAJECTORY_CONFIG_PATH);
  config.load(mc_rtc::Configuration::fromYAMLData("ObserverPipelines:\n"
                                                  "- name: RangerTrajectoryPipeline\n"
                                                  "  gui: false\n"
                                                  "  observers:\n"
                                                  "  - type: Encoder\n"
                                                  "    update: true\n"
                                                  "    position: control\n"
                                                  "    velocity: control\n"
                                                  "  - type: BodySensor\n"
                                                  "    update: true\n"
                                                  "    bodySensor: FloatingBase\n"
                                                  "    method: sensor\n"
                                                  "    updatePose: true\n"
                                                  "    updateVel: true\n"));
  return config;
}

std::unique_ptr<ProbeController> makeController(const std::function<void(mc_rtc::Configuration &)> & tweak = {})
{
  loadObserverModules();
  auto config = sampleConfiguration();
  if(tweak) { tweak(config); }
  auto controller =
      std::make_unique<ProbeController>(robotModule(), dt, config, mc_control::MCController::Backend::Tasks);
  controller->createObserverPipelines(config);
  controller->reset({controller->robot().mbc().q});
  return controller;
}

/** One cycle of the sample as MCGlobalController::run() plays it: observer
 * pipelines first, then the controller (which runs the FSM executor and the QP).
 */
struct Sample
{
  std::string state;
  Eigen::Vector2d achieved = Eigen::Vector2d::Zero();
  Eigen::Vector2d reference = Eigen::Vector2d::Zero();
  double curveTime = 0.0;
  bool hasReference = false;
  double z = 0.0;
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  double maxSteering = 0.0;
  double maxDriveTorque = 0.0;
  double minNormalForce = std::numeric_limits<double>::infinity();
  bool solved = false;
};

mc_tasks::BSplineTrajectoryTask * curveTask(const ProbeController & controller)
{
  // QPSolver::tasks() hands out non-const MetaTask pointers even from a const
  // solver, which is what lets the reference be sampled below.
  for(auto * task : controller.solver().tasks())
  {
    if(auto * spline = dynamic_cast<mc_tasks::BSplineTrajectoryTask *>(task)) { return spline; }
  }
  return nullptr;
}

const mc_solver::RollingContactDynamicsConstraint * rollingDynamics(const ProbeController & controller)
{
  for(const auto & constraint : controller.constraints_)
  {
    if(const auto * rolling = dynamic_cast<const mc_solver::RollingContactDynamicsConstraint *>(constraint.get()))
    {
      return rolling;
    }
  }
  return nullptr;
}

std::vector<Sample> rollout(ProbeController & controller, size_t cycles)
{
  const auto * rolling = rollingDynamics(controller);
  BOOST_REQUIRE(rolling != nullptr);
  std::vector<Sample> samples;
  samples.reserve(cycles);
  for(size_t i = 0; i < cycles; ++i)
  {
    controller.runObserverPipelines();
    Sample sample;
    sample.solved = controller.run();
    sample.state = controller.executor_.state();
    const auto & pose = controller.robot().frame("chassis").position();
    sample.achieved = pose.translation().head<2>();
    sample.z = pose.translation().z();
    // sva stores the world-to-body map, so the body-to-world attitude is its
    // transpose; RPY of that is what "the chassis attitude" means here.
    const Eigen::Matrix3d attitude = pose.rotation().transpose();
    sample.yaw = std::atan2(attitude(1, 0), attitude(0, 0));
    sample.pitch = std::asin(std::clamp(-attitude(2, 0), -1.0, 1.0));
    sample.roll = std::atan2(attitude(2, 1), attitude(2, 2));
    if(auto * spline = curveTask(controller))
    {
      // SplineTrajectoryTask::refPose() is protected; the curve itself is not,
      // and evaluating it at the task's own current time gives exactly the
      // reference the QP was handed on this cycle.
      sample.curveTime = std::min(spline->currentTime(), spline->duration());
      sample.reference = spline->spline().splev(sample.curveTime, 0)[0].head<2>();
      sample.hasReference = true;
    }
    for(const auto & wheel : wheelNames)
    {
      const auto steering = controller.robot().jointIndexByName(wheel + "_steer");
      const auto drive = controller.robot().jointIndexByName(wheel + "_drive");
      sample.maxSteering = std::max(sample.maxSteering, std::abs(controller.robot().mbc().q[steering][0]));
      sample.maxDriveTorque = std::max(sample.maxDriveTorque, std::abs(controller.robot().jointTorque()[drive][0]));
    }
    if(sample.solved)
    {
      // Each wheel owns eight generator multipliers inside the solver's global
      // lambda vector; normalForce() wants exactly that slice, as
      // MCRollingContactController::run() does it.
      auto & tasksSolver = static_cast<mc_solver::TasksQPSolver &>(controller.solver());
      const auto & lambda = tasksSolver.solver().lambdaVec();
      for(const auto & wheel : wheelNames)
      {
        const int begin = rolling->lambdaBegin(wheel) - tasksSolver.data().lambdaBegin();
        BOOST_REQUIRE(begin >= 0);
        BOOST_REQUIRE(begin + rolling->lambdaCount(wheel) <= lambda.size());
        sample.minNormalForce = std::min(sample.minNormalForce, rolling->normalForce(wheel, lambda.segment(begin, 8)));
      }
    }
    samples.push_back(std::move(sample));
  }
  return samples;
}

/** Ordered, de-duplicated list of the states the executor visited. */
std::vector<std::string> visitedStates(const std::vector<Sample> & samples)
{
  std::vector<std::string> states;
  for(const auto & sample : samples)
  {
    if(states.empty() || states.back() != sample.state) { states.push_back(sample.state); }
  }
  return states;
}

/** Index range [begin, end) of the samples belonging to one state. */
std::pair<size_t, size_t> stateRange(const std::vector<Sample> & samples, const std::string & state)
{
  const auto begin =
      std::find_if(samples.begin(), samples.end(), [&state](const Sample & s) { return s.state == state; });
  BOOST_REQUIRE(begin != samples.end());
  const auto end = std::find_if(begin, samples.end(), [&state](const Sample & s) { return s.state != state; });
  return {static_cast<size_t>(std::distance(samples.begin(), begin)),
          static_cast<size_t>(std::distance(samples.begin(), end))};
}

/** The three analytic curves the generator script fits, expressed as the
 * displacement from the curve's own start so the comparison does not depend on
 * where the previous state left the chassis (BSplineTrajectoryTask anchors its
 * curve at the frame position when the state starts).
 */
Eigen::Vector2d analyticCircle(double s)
{
  const double angle = 2.0 * M_PI * s;
  return {circleRadius * std::sin(angle), circleRadius * (1.0 - std::cos(angle))};
}

Eigen::Vector2d analyticLine(double s)
{
  return {lineLength * s, 0.0};
}

Eigen::Vector2d analyticLissajous(double s)
{
  const double theta0 = M_PI / 4.0; // pi / (2 * b) with b = 2
  const double phase = -3.0 * theta0; // -a * theta0 with a = 3
  const double theta = theta0 + 2.0 * M_PI * s;
  return {lissajousA * (std::sin(3.0 * theta + phase) - std::sin(3.0 * theta0 + phase)),
          lissajousB * (std::sin(2.0 * theta) - std::sin(2.0 * theta0))};
}

/** Worst distance between the reference the task actually generates and the
 * curve the generator script says it should be, both taken relative to the
 * state's own first reference sample.
 */
double referenceGeometryError(const std::vector<Sample> & samples,
                              const std::string & state,
                              double duration,
                              const std::function<Eigen::Vector2d(double)> & analytic)
{
  const auto range = stateRange(samples, state);
  BOOST_REQUIRE(samples[range.first].hasReference);
  // Both sequences are taken relative to the state's first recorded sample, so
  // neither the chassis position the previous state left behind nor the fact
  // that the first sample is already one dt into the curve biases the result.
  const Eigen::Vector2d origin = samples[range.first].reference;
  const Eigen::Vector2d analyticOrigin = analytic(std::min(1.0, samples[range.first].curveTime / duration));
  double worst = 0.0;
  for(size_t i = range.first; i < range.second; ++i)
  {
    if(!samples[i].hasReference) { continue; }
    const double s = std::min(1.0, samples[i].curveTime / duration);
    worst = std::max(worst, ((samples[i].reference - origin) - (analytic(s) - analyticOrigin)).norm());
  }
  return worst;
}

double pathLength(const std::vector<Sample> & samples, size_t begin, size_t end)
{
  double length = 0.0;
  for(size_t i = begin + 1; i < end; ++i) { length += (samples[i].achieved - samples[i - 1].achieved).norm(); }
  return length;
}

/** Turns off the eight predicted-rate rows.
 *
 * The sample ships with them on because that is the four-steering configuration
 * this project validated, but nothing outside C++ can fill in the references
 * they track, so they hold every wheel rate at zero -- see the KNOWN LIMITATION
 * block in the sample's YAML. Tests that need the chassis to actually move pass
 * this.
 */
void withoutRateRows(mc_rtc::Configuration & config)
{
  for(auto constraint : config("constraints"))
  {
    if(static_cast<std::string>(constraint("type")) == "rollingContact")
    {
      constraint.add("trackRotatingRates", false);
    }
  }
}

size_t fullSequenceCycles()
{
  return static_cast<size_t>((circleDuration + lineDuration + lissajousDuration) / dt) + 10;
}

} // namespace

/** The FSM plays the three curve states in order, each for exactly its
 * configured duration, and hands over on the timeElapsed completion criteria.
 */
BOOST_AUTO_TEST_CASE(FsmPlaysTheThreeCurveStatesInOrder)
{
  auto controller = makeController();
  const auto samples = rollout(*controller, fullSequenceCycles());

  const std::vector<std::string> expected = {"Ranger::Circle", "Ranger::Line", "Ranger::Lissajous"};
  const auto visited = visitedStates(samples);
  BOOST_REQUIRE_EQUAL(visited.size(), expected.size());
  BOOST_CHECK_EQUAL_COLLECTIONS(visited.begin(), visited.end(), expected.begin(), expected.end());

  const auto circle = stateRange(samples, "Ranger::Circle");
  const auto line = stateRange(samples, "Ranger::Line");
  const auto lissajous = stateRange(samples, "Ranger::Lissajous");
  BOOST_TEST_MESSAGE("state ranges: circle [" << circle.first << ", " << circle.second << ") line [" << line.first
                                              << ", " << line.second << ") lissajous [" << lissajous.first << ", "
                                              << lissajous.second << ")");
  BOOST_CHECK_EQUAL(circle.first, 0u);
  BOOST_CHECK_EQUAL(line.first, circle.second);
  BOOST_CHECK_EQUAL(lissajous.first, line.second);
  // Each state owns duration/dt cycles: its first run() advances the curve to
  // t = dt and the run() that reaches t = duration is its last. One extra cycle
  // is possible because currTime_ accumulates dt, and 900 * 0.005 lands just
  // below 4.5 in binary floating point (the line state does take 901).
  BOOST_CHECK_EQUAL(circle.second - circle.first, static_cast<size_t>(circleDuration / dt));
  BOOST_CHECK_EQUAL(line.second - line.first, static_cast<size_t>(lineDuration / dt) + 1);
  BOOST_CHECK_GE(lissajous.second - lissajous.first, static_cast<size_t>(lissajousDuration / dt));
}

/** The control points committed by make-trajectory-waypoints.py really do
 * produce a unit circle, a 2 m straight segment and a 3:2 Lissajous figure.
 *
 * This checks the generated geometry, not the robot: it compares the reference
 * BSplineTrajectoryTask emits against the analytic curves the script fits, and
 * is the assertion that fails if a controlPoints list is dropped, truncated or
 * mis-scaled.
 *
 * It runs without the rate rows because `controlPoints` are absolute world
 * points while the curve's start is wherever the frame is when the state begins
 * (mc_trajectory/BSpline.cpp). With the rate rows on the chassis never leaves
 * the origin, so the Lissajous curve would be anchored 2 m away from the point
 * its control points were generated for and would be genuinely deformed -- a
 * direct demonstration of why the generator asserts curve-to-curve continuity.
 */
BOOST_AUTO_TEST_CASE(GeneratedControlPointsReproduceTheIntendedCurves)
{
  auto controller = makeController(withoutRateRows);
  const auto samples = rollout(*controller, fullSequenceCycles());

  // The Bezier fit errors reported by the generator are 2.0e-8 m, 4.0e-15 m and
  // 2.0e-7 m; 5 mm leaves three orders of magnitude of headroom over that and
  // still fails on any real change to the control points. It is not tighter
  // because the anchor of each curve is the chassis position the previous state
  // left behind, which is a few millimetres off the nominal one.
  const double tolerance = 5e-3;
  BOOST_CHECK_LT(referenceGeometryError(samples, "Ranger::Circle", circleDuration, analyticCircle), tolerance);
  BOOST_CHECK_LT(referenceGeometryError(samples, "Ranger::Line", lineDuration, analyticLine), tolerance);
  BOOST_CHECK_LT(referenceGeometryError(samples, "Ranger::Lissajous", lissajousDuration, analyticLissajous), tolerance);

  // Continuity: each curve must start where the previous one ended, otherwise
  // the transition steps the reference and kicks the robot.
  const auto circle = stateRange(samples, "Ranger::Circle");
  const auto line = stateRange(samples, "Ranger::Line");
  const auto lissajous = stateRange(samples, "Ranger::Lissajous");
  BOOST_CHECK_LT((samples[line.first].reference - samples[circle.second - 1].reference).norm(), 1e-2);
  BOOST_CHECK_LT((samples[lissajous.first].reference - samples[line.second - 1].reference).norm(), 1e-2);
}

/** dimWeight zeroes roll, pitch and z, the constant-heading design holds yaw,
 * the QP solves every cycle, no wheel leaves the ground and the drive torque
 * stays inside the URDF limit.
 */
BOOST_AUTO_TEST_CASE(ChassisHoldsTheNonPlanarDegreesOfFreedomAndStaysWithinTheActuatorLimits)
{
  // Both configurations: as shipped (the rate rows hold every wheel still) and
  // with them released, where the chassis really does roll two metres per
  // straight segment and the drive torque has something to do.
  for(const bool rateRows : {true, false})
  {
    auto controller = rateRows ? makeController() : makeController(withoutRateRows);
    const auto samples = rollout(*controller, fullSequenceCycles());

    double worstZ = 0.0;
    double worstRoll = 0.0;
    double worstPitch = 0.0;
    double worstYaw = 0.0;
    double worstTorque = 0.0;
    double leastNormal = std::numeric_limits<double>::infinity();
    for(const auto & sample : samples)
    {
      BOOST_REQUIRE(sample.solved);
      worstZ = std::max(worstZ, std::abs(sample.z - 0.16));
      worstRoll = std::max(worstRoll, std::abs(sample.roll));
      worstPitch = std::max(worstPitch, std::abs(sample.pitch));
      worstYaw = std::max(worstYaw, std::abs(sample.yaw));
      worstTorque = std::max(worstTorque, sample.maxDriveTorque);
      leastNormal = std::min(leastNormal, sample.minNormalForce);
    }
    BOOST_TEST_MESSAGE("invariants (rate rows " << (rateRows ? "on" : "off") << "): z " << worstZ << " m, roll "
                                                << worstRoll << " rad, pitch " << worstPitch << " rad, yaw " << worstYaw
                                                << " rad, drive torque " << worstTorque << " Nm, least normal force "
                                                << leastNormal << " N");
    // dimWeight zeroes roll, pitch and z, so the rolling constraints own them
    // unopposed and hold them exactly.
    BOOST_CHECK_LT(worstZ, 1e-5);
    BOOST_CHECK_LT(worstRoll, 1e-5);
    BOOST_CHECK_LT(worstPitch, 1e-5);
    // The constant-heading design: every curve targets the reset orientation.
    BOOST_CHECK_LT(worstYaw, 1e-4);
    BOOST_CHECK_LT(worstTorque, driveTorqueLimit);
    // A wheel whose contact normal force reaches zero has left the ground.
    BOOST_CHECK_GT(leastNormal, 1.0);
  }
}

/** Characterises the two limitations documented at the top of the sample's
 * YAML. These are NOT quality bars: they pin a diagnosed shortcoming so it
 * cannot be forgotten, and they are expected to fail (and to be rewritten)
 * whenever the sample gains a way to command the steering hinges.
 */
BOOST_AUTO_TEST_CASE(RateRowsWithoutAReferenceProviderBrakeTheChassis)
{
  auto controller = makeController();
  const auto samples = rollout(*controller, fullSequenceCycles());

  // trackRotatingRates is on and nothing calls rotatingRateReference(), so the
  // eight rate rows track zero at an authority equivalent to a task weight of
  // 1000 and the chassis barely moves: 0.13 m over the whole 52.5 s sequence.
  BOOST_CHECK_LT(pathLength(samples, 0, samples.size()), 0.30);
  // With the drive hinges braked the steering hinges never move either.
  const double worstSteering = std::max_element(samples.begin(), samples.end(), [](const Sample & a, const Sample & b)
                                                { return a.maxSteering < b.maxSteering; })
                                   ->maxSteering;
  BOOST_TEST_MESSAGE("braked rollout: path " << pathLength(samples, 0, samples.size()) << " m, worst steering "
                                             << worstSteering << " rad");
  BOOST_CHECK_LT(worstSteering, 1e-4);
}

/** With the rate rows released the trajectory task and the rolling constraints
 * work exactly as intended along the wheels' own direction, and not at all
 * across it: the steering hinges have a zero column in every rolling row (the
 * scrub radius is zero), so no declarative task can turn them.
 */
BOOST_AUTO_TEST_CASE(WithoutRateRowsTheChassisTracksOnlyTheWheelAlignedAxis)
{
  auto controller = makeController(withoutRateRows);
  const auto samples = rollout(*controller, fullSequenceCycles());

  double worstAligned = 0.0;
  double worstTransverse = 0.0;
  double worstSteering = 0.0;
  for(const auto & sample : samples)
  {
    if(!sample.hasReference) { continue; }
    worstAligned = std::max(worstAligned, std::abs(sample.achieved.x() - sample.reference.x()));
    worstTransverse = std::max(worstTransverse, std::abs(sample.achieved.y()));
    worstSteering = std::max(worstSteering, sample.maxSteering);
  }
  BOOST_TEST_MESSAGE("released rollout: worst aligned " << worstAligned << " m, worst transverse " << worstTransverse
                                                        << " m, worst steering " << worstSteering << " rad");
  // Along +x -- the direction the wheels point at reset -- the chassis follows
  // every reference, curved states included, to under 20 mm (measured 16.3 mm,
  // during the fastest part of the Lissajous).
  BOOST_CHECK_LT(worstAligned, 2e-2);
  // Across it the soft lateral rows pin the chassis to its start line: 2.5 mm
  // over the whole sequence, against references that ask for 2 m of y.
  BOOST_CHECK_LT(worstTransverse, 1e-2);
  // And the hinges still never turn: 1.7e-5 rad of numerical drift over 10510
  // cycles, against a demand that sweeps a full turn.
  BOOST_CHECK_LT(worstSteering, 1e-4);
  // The straight-line state is the one curve the sample does follow; assert it
  // tightly so a regression in the trajectory task or the rolling rows shows up.
  const auto line = stateRange(samples, "Ranger::Line");
  double worstLine = 0.0;
  for(size_t i = line.first; i < line.second; ++i)
  {
    if(!samples[i].hasReference) { continue; }
    worstLine = std::max(worstLine, (samples[i].achieved - samples[i].reference).norm());
  }
  BOOST_CHECK_LT(worstLine, 5e-3);
  BOOST_CHECK_GT(pathLength(samples, line.first, line.second), 1.99);
}
