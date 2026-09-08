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
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace
{

constexpr double dt = 0.005;
constexpr double circleDuration = 21.0;
constexpr double lineDuration = 6.75;
constexpr double lissajousDuration = 51.0;
constexpr double circleRadius = 1.0;
constexpr double lineLength = 2.0;
constexpr double lissajousA = 1.0;
constexpr double lissajousB = 1.0;
/** URDF effort limit of the four *_drive joints. */
constexpr double driveTorqueLimit = 35.0;

const std::vector<std::string> wheelNames = {"front_left", "front_right", "rear_left", "rear_right"};
/** Keys of the four carrier steering tasks in the sample's `tasks` map. */
const std::vector<std::string> steeringTaskNames = {"FrontLeftSteering", "FrontRightSteering", "RearLeftSteering",
                                                    "RearRightSteering"};

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
  std::array<double, 4> steering = {0.0, 0.0, 0.0, 0.0};
  std::array<double, 4> steeringRate = {0.0, 0.0, 0.0, 0.0};
  std::array<double, 4> driveRate = {0.0, 0.0, 0.0, 0.0};
  double maxDriveTorque = 0.0;
  double minNormalForce = std::numeric_limits<double>::infinity();
  bool solved = false;

  double maxSteering() const
  {
    double worst = 0.0;
    for(double angle : steering) { worst = std::max(worst, std::abs(angle)); }
    return worst;
  }
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
    for(size_t w = 0; w < wheelNames.size(); ++w)
    {
      const auto steering = controller.robot().jointIndexByName(wheelNames[w] + "_steer");
      const auto drive = controller.robot().jointIndexByName(wheelNames[w] + "_drive");
      sample.steering[w] = controller.robot().mbc().q[steering][0];
      sample.steeringRate[w] = controller.robot().mbc().alpha[steering][0];
      sample.driveRate[w] = controller.robot().mbc().alpha[drive][0];
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

/** Switches the predicted-rate rows back on.
 *
 * The sample ships with them off because nothing outside C++ can fill in the
 * references they track (RollingContactConstraint::rotatingRateReference), so
 * they default to zero and become a "hold the wheels still" objective. One test
 * characterises that, so it needs a way to turn them back on.
 */
void withRateRows(mc_rtc::Configuration & config)
{
  for(auto constraint : config("constraints"))
  {
    if(static_cast<std::string>(constraint("type")) == "rollingContact") { constraint.add("trackRotatingRates", true); }
  }
}

/** Removes the four carrier steering tasks from every curve state.
 *
 * They have to go from the base state and from each generated state, because a
 * generated entry left behind would carry a schedule with no `type` and fail to
 * load. What is left is exactly the configuration that could not steer.
 */
void withoutSteeringTasks(mc_rtc::Configuration & config)
{
  auto states = config("states");
  for(const auto & name : states.keys())
  {
    auto state = states(name);
    if(!state.has("tasks")) { continue; }
    auto tasks = state("tasks");
    for(const auto & wheel : steeringTaskNames) { tasks.remove(wheel); }
  }
}

size_t fullSequenceCycles()
{
  return static_cast<size_t>((circleDuration + lineDuration + lissajousDuration) / dt) + 10;
}

/** The three configurations the assertions below need. */
enum class Variant
{
  /** The sample exactly as it ships. */
  Shipped,
  /** ... with the predicted-rate rows switched back on. */
  RateRows,
  /** ... with the four carrier steering tasks removed. */
  NoSteering
};

/** Roll out the whole sequence once per variant and cache it.
 *
 * Every assertion below reads the same trajectories, and a rollout is 15760
 * cycles of a real QP; running one per test case would cost several times more
 * for no extra coverage.
 */
const std::vector<Sample> & rollout(Variant variant)
{
  static std::map<Variant, std::vector<Sample>> cache;
  auto & samples = cache[variant];
  if(samples.empty())
  {
    std::function<void(mc_rtc::Configuration &)> tweak;
    if(variant == Variant::RateRows) { tweak = withRateRows; }
    if(variant == Variant::NoSteering) { tweak = withoutSteeringTasks; }
    auto controller = makeController(tweak);
    samples = rollout(*controller, fullSequenceCycles());
  }
  return samples;
}

const std::vector<Sample> & nominalRollout()
{
  return rollout(Variant::Shipped);
}

/** Chassis planar velocity by central difference of the recorded poses. */
Eigen::Vector2d chassisVelocity(const std::vector<Sample> & samples, size_t i)
{
  const size_t a = i ? i - 1 : i;
  const size_t b = std::min(samples.size() - 1, i + 1);
  if(b == a) { return Eigen::Vector2d::Zero(); }
  return (samples[b].achieved - samples[a].achieved) / (static_cast<double>(b - a) * dt);
}

/** Angle between the direction of travel and the wheel line, in [0, pi/2].
 *
 * The wheel is symmetric, so only the line matters: a wheel rolling backwards
 * along the right line is perfectly aligned.
 */
double wheelMisalignment(const std::vector<Sample> & samples, size_t i)
{
  const Eigen::Vector2d velocity = chassisVelocity(samples, i);
  const double travel = std::atan2(velocity.y(), velocity.x());
  double error = std::fmod(travel - samples[i].steering[0] + 0.5 * M_PI, M_PI);
  if(error < 0.0) { error += M_PI; }
  return std::abs(error - 0.5 * M_PI);
}

/** Index ranges over which a hinge is slewing through a branch flip.
 *
 * A flip is the only thing in the schedule that asks a hinge for more than
 * 3 rad/s: outside them the demand peaks near 1 rad/s (the Lissajous) and is
 * 0.30 rad/s on the circle, while a flip is scheduled at pi / 0.6 = 5.24 rad/s.
 */
std::vector<std::pair<size_t, size_t>> branchFlips(const std::vector<Sample> & samples)
{
  const double slewing = 3.0;
  std::vector<std::pair<size_t, size_t>> flips;
  for(size_t i = 0; i < samples.size(); ++i)
  {
    if(std::abs(samples[i].steeringRate[0]) < slewing) { continue; }
    size_t end = i;
    while(end + 1 < samples.size() && std::abs(samples[end + 1].steeringRate[0]) >= slewing) { ++end; }
    flips.emplace_back(i, end);
    i = end;
  }
  return flips;
}

} // namespace

/** The FSM plays the three curve states in order, each for exactly its
 * configured duration, and hands over on the timeElapsed completion criteria.
 */
BOOST_AUTO_TEST_CASE(FsmPlaysTheThreeCurveStatesInOrder)
{
  const auto & samples = nominalRollout();

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
  // is possible because currTime_ accumulates dt and the accumulated sum can
  // land just below the duration in binary floating point.
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
 */
BOOST_AUTO_TEST_CASE(GeneratedControlPointsReproduceTheIntendedCurves)
{
  const auto & samples = nominalRollout();

  // The Bezier fit errors reported by the generator are 2.0e-8 m, 4.0e-15 m and
  // 2.0e-7 m. The bound is far looser than that because `controlPoints` are
  // absolute world points while the curve is anchored at wherever the frame is
  // when the state starts (src/mc_trajectory/BSpline.cpp): each curve therefore
  // inherits, and slightly deforms around, the centimetre of tracking error the
  // previous state left behind.
  const double tolerance = 2e-2;
  const double circleError = referenceGeometryError(samples, "Ranger::Circle", circleDuration, analyticCircle);
  const double lineError = referenceGeometryError(samples, "Ranger::Line", lineDuration, analyticLine);
  const double lissajousError =
      referenceGeometryError(samples, "Ranger::Lissajous", lissajousDuration, analyticLissajous);
  BOOST_TEST_MESSAGE("reference geometry error: circle " << circleError << " m, line " << lineError << " m, lissajous "
                                                         << lissajousError << " m");
  BOOST_CHECK_LT(circleError, tolerance);
  BOOST_CHECK_LT(lineError, tolerance);
  BOOST_CHECK_LT(lissajousError, tolerance);

  // Continuity: each curve must start where the previous one ended, otherwise
  // the transition steps the reference and kicks the robot.
  const auto circle = stateRange(samples, "Ranger::Circle");
  const auto line = stateRange(samples, "Ranger::Line");
  const auto lissajous = stateRange(samples, "Ranger::Lissajous");
  BOOST_CHECK_LT((samples[line.first].reference - samples[circle.second - 1].reference).norm(), 2e-2);
  BOOST_CHECK_LT((samples[lissajous.first].reference - samples[line.second - 1].reference).norm(), 2e-2);
}

/** The four hinges follow one shared schedule and stay inside their limit.
 *
 * The generator emits a single waypoint list per curve and the YAML aliases it
 * to all four carriers, on the grounds that a constant-heading chassis
 * translates and every wheel centre therefore sees the same velocity. If that
 * ever stops being true the four hinges will diverge here.
 */
BOOST_AUTO_TEST_CASE(AllFourHingesFollowOneSharedSchedule)
{
  const auto & samples = nominalRollout();

  double spread = 0.0;
  double worst = 0.0;
  for(const auto & sample : samples)
  {
    for(size_t w = 1; w < sample.steering.size(); ++w)
    {
      spread = std::max(spread, std::abs(sample.steering[w] - sample.steering[0]));
    }
    worst = std::max(worst, sample.maxSteering());
  }
  BOOST_TEST_MESSAGE("hinge spread " << spread << " rad, worst |delta| " << worst << " rad");
  BOOST_CHECK_LT(spread, 1e-9);
  // The URDF limit; the schedule is built to approach it but never ask for more.
  BOOST_CHECK_LT(worst, 0.5 * M_PI);
  BOOST_CHECK_GT(worst, 1.4);

  // Per state: the curved states sweep the hinges nearly to the limit, the
  // straight one leaves them alone.
  for(const auto & entry :
      {std::make_pair(std::string("Ranger::Circle"), true), std::make_pair(std::string("Ranger::Line"), false),
       std::make_pair(std::string("Ranger::Lissajous"), true)})
  {
    const auto range = stateRange(samples, entry.first);
    double low = 0.0;
    double high = 0.0;
    for(size_t i = range.first; i < range.second; ++i)
    {
      low = std::min(low, samples[i].steering[0]);
      high = std::max(high, samples[i].steering[0]);
    }
    BOOST_TEST_MESSAGE(entry.first << " hinge span [" << low << ", " << high << "] rad");
    if(entry.second)
    {
      BOOST_CHECK_LT(low, -1.4);
      BOOST_CHECK_GT(high, 1.4);
    }
    else
    {
      BOOST_CHECK_LT(high - low, 0.05);
    }
  }
}

/** The wheels really do point along the direction of travel.
 *
 * This is a consistency check between the robot and itself, independent of the
 * generator: whenever the chassis is moving, the angle between its velocity and
 * the wheel line has to be small. It catches a schedule aliased to the wrong
 * carrier frame (the wheels then disagree and the chassis barely moves, so the
 * sample count collapses) and it degrades on a badly interpolated one. It is
 * deliberately NOT the check for a *sign* error: the chassis follows wherever
 * its wheels point, so a mirrored schedule still looks locally aligned -- that
 * one is caught by ChassisFollowsEachCurve, where it shows up as a metre.
 */
BOOST_AUTO_TEST_CASE(WheelsPointAlongTheDirectionOfTravel)
{
  const auto & samples = nominalRollout();

  std::vector<double> misalignment;
  for(size_t i = 0; i < samples.size(); ++i)
  {
    if(!samples[i].hasReference) { continue; }
    // Below this speed the lateral rows are satisfied whatever the hinges do,
    // and the direction of travel is not meaningfully defined; the Lissajous
    // has two genuine cusps where the chassis stops dead.
    if(chassisVelocity(samples, i).norm() < 0.05) { continue; }
    misalignment.push_back(wheelMisalignment(samples, i));
  }
  BOOST_REQUIRE_GT(misalignment.size(), samples.size() / 2);
  std::sort(misalignment.begin(), misalignment.end());
  const double median = misalignment[misalignment.size() / 2];
  const double p99 = misalignment[(misalignment.size() * 99) / 100];
  const double worst = misalignment.back();
  BOOST_TEST_MESSAGE("wheel misalignment: median " << median << " rad, p99 " << p99 << " rad, max " << worst << " rad");
  // Measured: median 0.0019-0.0038 rad (0.11-0.22 deg), p99 0.0097 rad
  // (0.56 deg), max 0.030 rad (1.74 deg, inside a branch flip).
  BOOST_CHECK_LT(median, 0.01);
  BOOST_CHECK_LT(p99, 0.02);
  BOOST_CHECK_LT(worst, 0.06);
}

/** Every branch flip reverses the drive direction, with nothing commanding it.
 *
 * The schedule only ever tells a hinge where to point. After a flip the wheel
 * faces 180 deg the other way, so it has to roll backwards to keep the chassis
 * going the same way -- and the only thing that can decide that is the rolling
 * row coupling the wheel rate to the chassis motion. Six flips are scheduled:
 * two on the circle, four on the Lissajous.
 */
BOOST_AUTO_TEST_CASE(DriveDirectionReversesAcrossEveryBranchFlip)
{
  const auto & samples = nominalRollout();
  const auto flips = branchFlips(samples);

  BOOST_TEST_MESSAGE("branch flips detected: " << flips.size());
  BOOST_REQUIRE_EQUAL(flips.size(), 6u);
  const size_t margin = static_cast<size_t>(0.5 / dt);
  for(const auto & flip : flips)
  {
    BOOST_REQUIRE_GE(flip.first, margin);
    BOOST_REQUIRE_LT(flip.second + margin, samples.size());
    const double before = samples[flip.first - margin].driveRate[0];
    const double after = samples[flip.second + margin].driveRate[0];
    BOOST_TEST_MESSAGE("  flip at t = " << static_cast<double>(flip.first) * dt << " s: drive rate " << before << " -> "
                                        << after << " rad/s");
    BOOST_CHECK_GT(std::abs(before), 1.0);
    BOOST_CHECK_GT(std::abs(after), 1.0);
    BOOST_CHECK_LT(before * after, 0.0);
  }
}

/** The chassis follows each curve. */
BOOST_AUTO_TEST_CASE(ChassisFollowsEachCurve)
{
  const auto & samples = nominalRollout();

  // Bounds are roughly 1.4x the measured values, which are dominated by the
  // branch flips: a flip swings a hinge through 180 deg over 0.6 s, and while
  // it does the wheels are not pointing where the chassis needs to go. Between
  // flips the tracking is millimetric.
  const std::vector<std::tuple<std::string, double, double, double, double>> expectations = {
      // state, max deviation, mean deviation, commanded length, length tolerance
      {"Ranger::Circle", 0.13, 0.03, 6.283, 0.15},
      {"Ranger::Line", 0.015, 0.010, 2.000, 0.02},
      {"Ranger::Lissajous", 0.095, 0.016, 15.209, 0.40},
  };
  for(const auto & expectation : expectations)
  {
    const auto range = stateRange(samples, std::get<0>(expectation));
    double worst = 0.0;
    double total = 0.0;
    size_t count = 0;
    for(size_t i = range.first; i < range.second; ++i)
    {
      if(!samples[i].hasReference) { continue; }
      const double error = (samples[i].achieved - samples[i].reference).norm();
      worst = std::max(worst, error);
      total += error;
      ++count;
    }
    BOOST_REQUIRE_GT(count, 0u);
    const double mean = total / static_cast<double>(count);
    const double travelled = pathLength(samples, range.first, range.second);
    BOOST_TEST_MESSAGE(std::get<0>(expectation) << ": max deviation " << worst << " m, mean " << mean
                                                << " m, path travelled " << travelled << " m");
    BOOST_CHECK_LT(worst, std::get<1>(expectation));
    BOOST_CHECK_LT(mean, std::get<2>(expectation));
    BOOST_CHECK_LT(std::abs(travelled - std::get<3>(expectation)), std::get<4>(expectation));
  }
}

/** dimWeight zeroes roll, pitch and z, the constant-heading design holds yaw,
 * the QP solves every cycle, no wheel leaves the ground and the drive torque
 * stays inside the URDF limit.
 */
BOOST_AUTO_TEST_CASE(ChassisHoldsTheNonPlanarDegreesOfFreedomAndStaysWithinTheActuatorLimits)
{
  const auto & samples = nominalRollout();

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
  BOOST_TEST_MESSAGE("invariants: z " << worstZ << " m, roll " << worstRoll << " rad, pitch " << worstPitch
                                      << " rad, yaw " << worstYaw << " rad, drive torque " << worstTorque
                                      << " Nm, least normal force " << leastNormal << " N");
  // dimWeight zeroes roll, pitch and z, so the rolling constraints own them
  // unopposed and hold them exactly.
  BOOST_CHECK_LT(worstZ, 1e-5);
  BOOST_CHECK_LT(worstRoll, 1e-5);
  BOOST_CHECK_LT(worstPitch, 1e-5);
  // The constant-heading design. Not exact: the carrier orientation tasks act on
  // the chassis yaw as well as on their hinge, which is why their weight is 100
  // against the chassis task's 1000. Measured 5.5e-3 rad; at weight 500 it was
  // 2.7e-2 and at 4000 it was 0.11.
  BOOST_CHECK_LT(worstYaw, 1e-2);
  BOOST_CHECK_LT(worstTorque, driveTorqueLimit);
  // A wheel whose contact normal force reaches zero has left the ground.
  BOOST_CHECK_GT(leastNormal, 1.0);
}

/** Characterises why `trackRotatingRates` is off in the shipped configuration.
 *
 * Not a quality bar: it pins a documented property of the constraint so nobody
 * switches those rows on again without noticing what they do here. If a future
 * change gives the rate rows a reference provider this test should be deleted,
 * not relaxed.
 */
BOOST_AUTO_TEST_CASE(RateRowsWithoutAReferenceProviderBrakeTheChassis)
{
  const auto & samples = rollout(Variant::RateRows);

  // The eight rate rows track zero at an authority equivalent to a task weight
  // of 1000, because nothing calls rotatingRateReference(). Measured: 0.20 m of
  // travel over the whole sequence, against 23.6 m in the shipped configuration.
  const double travelled = pathLength(samples, 0, samples.size());
  double worstSteering = 0.0;
  for(const auto & sample : samples) { worstSteering = std::max(worstSteering, sample.maxSteering()); }
  BOOST_TEST_MESSAGE("braked rollout: path " << travelled << " m, worst |delta| " << worstSteering << " rad");
  BOOST_CHECK_LT(travelled, 0.5);
  BOOST_CHECK_LT(pathLength(nominalRollout(), 0, nominalRollout().size()), 25.0);
  BOOST_CHECK_GT(pathLength(nominalRollout(), 0, nominalRollout().size()), 20.0);
}

/** Characterises why the four carrier steering tasks have to exist.
 *
 * Remove them and nothing in the QP can turn a steering hinge: the steering
 * axis passes through the wheel centre, so its column in the contact-point
 * Jacobian is identically zero and the rolling, lateral and normal rows have no
 * coefficient on delta-ddot. The chassis is then confined to the line its
 * wheels started on, however the chassis trajectory task is weighted. This is
 * what the *Steering tasks buy, measured rather than argued.
 */
BOOST_AUTO_TEST_CASE(WithoutTheSteeringTasksTheChassisCannotLeaveItsWheelLine)
{
  const auto & samples = rollout(Variant::NoSteering);

  double worstSteering = 0.0;
  double worstTransverse = 0.0;
  double worstAligned = 0.0;
  for(const auto & sample : samples)
  {
    worstSteering = std::max(worstSteering, sample.maxSteering());
    if(!sample.hasReference) { continue; }
    worstTransverse = std::max(worstTransverse, std::abs(sample.achieved.y()));
    worstAligned = std::max(worstAligned, std::abs(sample.achieved.x() - sample.reference.x()));
  }
  BOOST_TEST_MESSAGE("unsteered rollout: worst |delta| " << worstSteering << " rad, worst |y| " << worstTransverse
                                                         << " m, worst x error " << worstAligned << " m");
  // The hinges never turn.
  BOOST_CHECK_LT(worstSteering, 1e-4);
  // The chassis stays on its start line even though the references ask for two
  // metres of y.
  BOOST_CHECK_LT(worstTransverse, 1e-2);
  // Along the wheels' own direction it still tracks every reference, which is
  // what isolates the failure to the steering degrees of freedom.
  BOOST_CHECK_LT(worstAligned, 2e-2);
  // And with the steering tasks in place it does leave that line.
  double reached = 0.0;
  for(const auto & sample : nominalRollout()) { reached = std::max(reached, std::abs(sample.achieved.y())); }
  BOOST_CHECK_GT(reached, 1.5);
}
