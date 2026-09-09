#include <mc_mujoco/mj_configuration.h>
#include <mc_mujoco/mj_sim.h>

#include <mc_rtc/DataStore.h>

#include <SpaceVecAlg/SpaceVecAlg>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/QR>

#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <utility>
#include <vector>

namespace
{

constexpr double pi = 3.14159265358979323846;
// Ranger's controller frame is at the centre of the chassis shell.  The
// steering/wheel-bearing bodies are 35 mm below that frame, so the root pose
// must be wheel-radius + 35 mm above the ground for the wheel contacts to be
// tangent.  Keep this convention in the deterministic runner as well as in
// the URDF and MuJoCo model.
constexpr double rangerChassisToAxle = 0.035;

struct Options
{
  std::string mcConfig;
  std::string robot;
  std::string scenario;
  std::string backend;
  std::string csv;
  std::string report;
  size_t cycles = 0;
  size_t warmupCycles = 100;
  double rampDegrees = 0.0;
  // Where on the terrain the chassis is reset, in terrain tangent coordinates.
  // Zero reproduces the historical reset at the terrain origin. The ramp
  // terrain (src/mc_robots/rolling_contact_description/mujoco/ramp_terrain.xml)
  // puts one ramp lane every 4 m in y, so `--start-y 8` is "reset in front of
  // the 15 degree ramp".
  double startX = 0.0;
  double startY = 0.0;
  double friction = 0.8;
  long frictionCycle = -1;
  double linearSpeed = 0.15;
  double yawRate = 0.25;
  double steeringAngle = 0.3;
  long impulseCycle = -1;
  double impulse = 0.0;
  std::string impulseDirection = "lateral";
  long separationCycle = -1;
  double separation = 0.0;
  // Three deliberate departures from what `mc_mujoco -f <config>` does, each
  // now an explicit switch instead of a hard-coded behaviour of this runner.
  // They all DEFAULT TO THE mc_mujoco BEHAVIOUR, so a case that wants the
  // easier regime has to say so in the case table and is visible there.
  //
  // torqueControl: mc_mujoco's MjConfiguration::torque_control defaults to
  //   false and the acceptance command does not pass --torque-control, so the
  //   Ranger's <motor> actuators are driven by mc_mujoco's joint PD on the
  //   controller's q/alpha output, not by the QP's joint torques. Running the
  //   suite exclusively with torque_control = true hid the whole PD path.
  // measuredContacts: calls RollingContact::SetMeasuredContact with MuJoCo's
  //   true contact forces and slips every substep. mc_mujoco does not, so the
  //   contact-mode estimator normally has no contact sensor at all. Feeding it
  //   ground truth structurally hides every defect in the sensorless path.
  // presetSteering: resets the four-steering scenarios with the hinges already
  //   at the manoeuvre's steady-state angle, which skips the steering transient
  //   entirely - and that transient is where the real failures happen.
  bool torqueControl = false;
  bool measuredContacts = false;
  bool presetSteering = false;
};

struct Wheel
{
  std::string name;
  std::string body;
  std::string drive;
  std::string steer;
  Eigen::Vector2d offset = Eigen::Vector2d::Zero();
  double radius = 0.2;
  int bodyId = -1;
  int geomId = -1;
  int driveJointId = -1;
  int steerJointId = -1;
  double previousQ = 0.0;
};

struct WheelSample
{
  double q = 0.0;
  double qd = 0.0;
  double totalSpinRate = 0.0;
  double steering = 0.0;
  double rollingSlip = 0.0;
  double lateralSlip = 0.0;
  double normalSpeed = 0.0;
  double normalForce = 0.0;
  double tangentialForce = 0.0;
  double frictionMargin = 0.0;
  double driveTorque = 0.0;
  double steerTorque = 0.0;
  double driveTarget = 0.0;
  double driveAcceleration = 0.0;
  double controllerDrivePosition = 0.0;
  double controllerDriveVelocity = 0.0;
  double activation = 0.0;
  double solverActivation = 0.0;
  double accelerationResidual = 0.0;
  double lateralAccelerationResidual = 0.0;
  double normalAccelerationResidual = 0.0;
  double hardPromotionResidual = 0.0;
  double qpNormalForce = 0.0;
  double qpTangentialForce = 0.0;
  // Angle between MuJoCo's actual contact normal under this wheel and the
  // terrain normal the controller was configured with, in radians, unsigned.
  // Standing assumption A4 of the formulation (`as:one-plane`) says all four
  // contacts share one plane and the controller takes that plane's normal as a
  // constant, so this number is the direct measure of how far a sample is from
  // the modelling assumption: 0 on flat ground and on a plateau, theta while a
  // wheel is on an incline the controller has not been told about. It is left
  // at 0 for a wheel with no contact, which `contacts` already reports.
  double contactNormalDeviation = 0.0;
  std::string mode = "unknown";
  int slidingGenerator = -1;
  int contacts = 0;
};

struct Statistics
{
  size_t samples = 0;
  size_t noContactSamples = 0;
  double maxRollingSlip = 0.0;
  double maxLateralSlip = 0.0;
  double maxNormalSpeed = 0.0;
  double minNormalForce = std::numeric_limits<double>::infinity();
  double minFrictionMargin = std::numeric_limits<double>::infinity();
  double maxDriveTorque = 0.0;
  double maxSteerTorque = 0.0;
  double maxContactNormalDeviation = 0.0;
  double maxTrajectoryError = 0.0;
  double trajectoryErrorSquared = 0.0;
  double maxYawError = 0.0;
  double maxTrackingPositionError = 0.0;
  double trackingPositionErrorSquared = 0.0;
  double maxTrackingYawError = 0.0;
  double trackingYawErrorSquared = 0.0;
  size_t missedDeadlines = 0;
  size_t contactFallbackSamples = 0;
  std::vector<double> rollingSlips;
  std::vector<double> lateralSlips;
  std::vector<double> normalSpeeds;
  std::vector<double> odometryPositionErrors;
  std::vector<double> odometryYawErrors;
  std::vector<double> trackingPositionErrors;
  std::vector<double> trackingYawErrors;
  std::vector<double> cycleWallMilliseconds;
  std::set<std::string> modes;
};

struct ControllerReference
{
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  double yaw = 0.0;
  double postureEvalNorm = 0.0;
  double positionEvalNorm = 0.0;
  double orientationEvalNorm = 0.0;
  double hardRhsNorm = 0.0;
  double rollingWeight = 0.0;
  double dynamicsResidual = 0.0;
  double floatingBaseEffortNorm = 0.0;
};

[[noreturn]] void usage(const std::string & reason)
{
  throw std::invalid_argument(
      reason
      + "\nusage: rolling_contact_mujoco_runner --mc-config FILE --robot differential|four-steering"
        " --backend Tasks|TVM"
        " --scenario NAME --cycles N --csv FILE --report FILE [--warmup-cycles N] [--ramp-deg D]"
        " [--start-x M] [--start-y M] [--friction MU]"
        " [--friction-cycle N]"
        " [--linear-speed MPS] [--yaw-rate RADPS] [--steering-angle RAD]"
        " [--impulse-cycle N --impulse MPS --impulse-direction longitudinal|lateral|normal]"
        " [--separation-cycle N --separation M]");
}

double parseDouble(const std::string & value, const std::string & option)
{
  size_t consumed = 0;
  const double out = std::stod(value, &consumed);
  if(consumed != value.size() || !std::isfinite(out)) { usage("invalid value for " + option + ": " + value); }
  return out;
}

bool parseBool(const std::string & value, const std::string & option)
{
  if(value == "true" || value == "1") { return true; }
  if(value == "false" || value == "0") { return false; }
  usage("invalid boolean for " + option + ": " + value);
  return false;
}

long parseLong(const std::string & value, const std::string & option)
{
  size_t consumed = 0;
  const long out = std::stol(value, &consumed);
  if(consumed != value.size()) { usage("invalid value for " + option + ": " + value); }
  return out;
}

Options parseOptions(int argc, char ** argv)
{
  Options out;
  for(int i = 1; i < argc; ++i)
  {
    const std::string option = argv[i];
    if(i + 1 >= argc) { usage("missing value for " + option); }
    const std::string value = argv[++i];
    if(option == "--mc-config") { out.mcConfig = value; }
    else if(option == "--robot") { out.robot = value; }
    else if(option == "--scenario") { out.scenario = value; }
    else if(option == "--backend") { out.backend = value; }
    else if(option == "--cycles")
    {
      const long cycles = parseLong(value, option);
      if(cycles <= 0) { usage("--cycles must be positive"); }
      out.cycles = static_cast<size_t>(cycles);
    }
    else if(option == "--csv") { out.csv = value; }
    else if(option == "--report") { out.report = value; }
    else if(option == "--warmup-cycles")
    {
      const long warmup = parseLong(value, option);
      if(warmup < 0) { usage("--warmup-cycles must be non-negative"); }
      out.warmupCycles = static_cast<size_t>(warmup);
    }
    else if(option == "--ramp-deg") { out.rampDegrees = parseDouble(value, option); }
    else if(option == "--start-x") { out.startX = parseDouble(value, option); }
    else if(option == "--start-y") { out.startY = parseDouble(value, option); }
    else if(option == "--friction") { out.friction = parseDouble(value, option); }
    else if(option == "--friction-cycle") { out.frictionCycle = parseLong(value, option); }
    else if(option == "--linear-speed") { out.linearSpeed = parseDouble(value, option); }
    else if(option == "--yaw-rate") { out.yawRate = parseDouble(value, option); }
    else if(option == "--steering-angle") { out.steeringAngle = parseDouble(value, option); }
    else if(option == "--impulse-cycle") { out.impulseCycle = parseLong(value, option); }
    else if(option == "--impulse") { out.impulse = parseDouble(value, option); }
    else if(option == "--impulse-direction") { out.impulseDirection = value; }
    else if(option == "--separation-cycle") { out.separationCycle = parseLong(value, option); }
    else if(option == "--separation") { out.separation = parseDouble(value, option); }
    else if(option == "--torque-control") { out.torqueControl = parseBool(value, option); }
    else if(option == "--measured-contacts") { out.measuredContacts = parseBool(value, option); }
    else if(option == "--preset-steering") { out.presetSteering = parseBool(value, option); }
    else { usage("unknown option: " + option); }
  }
  if(out.mcConfig.empty() || out.robot.empty() || out.scenario.empty() || out.backend.empty() || out.cycles == 0
     || out.csv.empty()
     || out.report.empty())
  {
    usage("missing required option");
  }
  if(out.robot != "differential" && out.robot != "four-steering") { usage("unsupported robot: " + out.robot); }
  if(out.backend != "Tasks" && out.backend != "TVM") { usage("unsupported backend: " + out.backend); }
  if(out.warmupCycles >= out.cycles) { usage("--warmup-cycles must be smaller than --cycles"); }
  if(out.friction <= 0.0) { usage("--friction must be positive"); }
  if(std::abs(out.rampDegrees) >= 45.0) { usage("absolute ramp angle must be below 45 degrees"); }
  if(std::abs(out.steeringAngle) > pi) { usage("absolute steering angle cannot exceed pi"); }
  if(out.impulseDirection != "longitudinal" && out.impulseDirection != "lateral"
     && out.impulseDirection != "normal")
  {
    usage("unsupported impulse direction: " + out.impulseDirection);
  }
  if(out.separation < 0.0) { usage("--separation must be non-negative"); }
  return out;
}

int namedId(const mjModel & model, int type, const std::string & name)
{
  const int id = mj_name2id(&model, type, name.c_str());
  if(id < 0) { throw std::runtime_error("MuJoCo object is missing: " + name); }
  return id;
}

Eigen::Vector3d column(const mjtNum * matrix, int index)
{
  return {matrix[index], matrix[3 + index], matrix[6 + index]};
}

double wrapAngle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

double rms(const std::vector<double> & values)
{
  if(values.empty()) { return 0.0; }
  double sum = 0.0;
  for(const double value : values) { sum += value * value; }
  return std::sqrt(sum / static_cast<double>(values.size()));
}

double percentile(std::vector<double> values, double probability)
{
  if(values.empty()) { return 0.0; }
  const double rank = std::ceil(probability * static_cast<double>(values.size())) - 1.0;
  const size_t index = std::min(values.size() - 1, static_cast<size_t>(rank));
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
  return values[index];
}

double yawInTerrain(const mjData & data,
                    int chassisBody,
                    const Eigen::Vector3d & tangentX,
                    const Eigen::Vector3d & tangentY)
{
  const Eigen::Vector3d bodyX = column(&data.xmat[9 * chassisBody], 0);
  return std::atan2(bodyX.dot(tangentY), bodyX.dot(tangentX));
}

std::vector<Wheel> makeWheels(const Options & options)
{
  if(options.robot == "differential")
  {
    return {{"left", "left_wheel", "left_drive", "", {0.0, 0.3}},
            {"right", "right_wheel", "right_drive", "", {0.0, -0.3}}};
  }
  return {{"front_left", "front_left_wheel", "front_left_drive", "front_left_steer", {0.247, 0.182}, 0.125},
          {"front_right", "front_right_wheel", "front_right_drive", "front_right_steer", {0.247, -0.182}, 0.125},
          {"rear_left", "rear_left_wheel", "rear_left_drive", "rear_left_steer", {-0.247, 0.182}, 0.125},
          {"rear_right", "rear_right_wheel", "rear_right_drive", "rear_right_steer", {-0.247, -0.182}, 0.125}};
}

bool close(double lhs, double rhs, double tolerance = 1e-12)
{
  return std::abs(lhs - rhs) <= tolerance * std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

void requireBody(const mjModel & model,
                 const std::string & name,
                 double mass,
                 const Eigen::Vector3d & centerOfMass,
                 const Eigen::Vector3d & inertia,
                 bool & staticParity)
{
  const int body = namedId(model, mjOBJ_BODY, name);
  staticParity = staticParity && close(model.body_mass[body], mass);
  for(int axis = 0; axis < 3; ++axis)
  {
    staticParity = staticParity && close(model.body_ipos[3 * body + axis], centerOfMass[axis]);
    staticParity = staticParity && close(model.body_inertia[3 * body + axis], inertia[axis]);
  }
}

bool checkStaticParity(const mjModel & model, const Options & options, const std::vector<Wheel> & wheels)
{
  const std::string prefix = options.robot == "differential" ? "rolling_diff_" : "ranger_mini_v3_";
  const size_t jointsPerWheel = options.robot == "differential" ? 1 : 2;
  bool parity = close(model.opt.timestep, 0.001)
                && model.njnt == static_cast<int>(jointsPerWheel * wheels.size() + 1)
                && model.nu == static_cast<int>(wheels.size() * (options.robot == "differential" ? 1 : 2));
  const int expectedNq = options.robot == "differential" ? 9 : 15;
  const int expectedNv = options.robot == "differential" ? 8 : 14;
  parity = parity && model.nq == expectedNq && model.nv == expectedNv;
  const int chassisBody = namedId(model, mjOBJ_BODY, prefix + "chassis");
  const double expectedChassisHeight = options.robot == "differential" ? 0.2 : 0.16;
  parity = parity && close(model.body_pos[3 * chassisBody], 0.0)
           && close(model.body_pos[3 * chassisBody + 1], 0.0)
           && close(model.body_pos[3 * chassisBody + 2], expectedChassisHeight);
  if(options.robot != "differential")
  {
    const int frameSite = namedId(model, mjOBJ_SITE, prefix + "chassis_frame");
    const int shellGeom = namedId(model, mjOBJ_GEOM, prefix + "chassis_shell_visual");
    const int collisionGeom = namedId(model, mjOBJ_GEOM, prefix + "chassis_collision");
    for(const int object : {frameSite, shellGeom, collisionGeom})
    {
      const auto bodyId = object == frameSite ? model.site_bodyid[object] : model.geom_bodyid[object];
      const auto position = object == frameSite ? model.site_pos + 3 * object : model.geom_pos + 3 * object;
      parity = parity && bodyId == chassisBody && close(position[0], 0.0) && close(position[1], 0.0)
               && close(position[2], 0.0);
    }
  }
  requireBody(model, prefix + "chassis", options.robot == "differential" ? 10.0 : 65.0,
              options.robot == "differential" ? Eigen::Vector3d{0.0, 0.0, -0.08}
                                                : Eigen::Vector3d::Zero(),
              options.robot == "differential" ? Eigen::Vector3d{0.32, 0.50, 0.62}
                                                : Eigen::Vector3d{1.03, 2.03, 2.91},
              parity);
  for(const auto & wheel : wheels)
  {
    requireBody(model, wheel.body, options.robot == "differential" ? 1.0 : 2.0, Eigen::Vector3d::Zero(),
                options.robot == "differential" ? Eigen::Vector3d{0.021, 0.040, 0.021}
                                                  : Eigen::Vector3d{0.0089, 0.015625, 0.0089},
                parity);
    const int joint = namedId(model, mjOBJ_JOINT, wheel.drive);
    parity = parity && model.jnt_type[joint] == mjJNT_HINGE;
    if(!wheel.steer.empty())
    {
      const std::string knuckle = prefix + wheel.name + "_knuckle";
      const int knuckleBody = namedId(model, mjOBJ_BODY, knuckle);
      parity = parity && close(model.body_pos[3 * knuckleBody], wheel.offset.x())
               && close(model.body_pos[3 * knuckleBody + 1], wheel.offset.y())
               && close(model.body_pos[3 * knuckleBody + 2], -rangerChassisToAxle);
      requireBody(model, knuckle, 0.5, Eigen::Vector3d::Zero(), {0.002, 0.002, 0.002}, parity);
      const int steering = namedId(model, mjOBJ_JOINT, wheel.steer);
      parity = parity && model.jnt_limited[steering] && close(model.jnt_range[2 * steering], -0.5 * pi, 2e-6)
               && close(model.jnt_range[2 * steering + 1], 0.5 * pi, 2e-6);
    }
  }
  std::vector<std::string> expectedJoints;
  for(const auto & wheel : wheels)
  {
    if(!wheel.steer.empty()) { expectedJoints.push_back(wheel.steer); }
    expectedJoints.push_back(wheel.drive);
  }
  if(model.njnt != static_cast<int>(expectedJoints.size() + 1)) { parity = false; }
  else
  {
    parity = parity && model.jnt_type[0] == mjJNT_FREE;
    for(size_t i = 0; i < expectedJoints.size(); ++i)
    {
      const char * name = mj_id2name(&model, mjOBJ_JOINT, static_cast<int>(i + 1));
      parity = parity && name && expectedJoints[i] == name;
    }
  }
  for(int actuator = 0; actuator < model.nu; ++actuator)
  {
    const int joint = model.actuator_trnid[2 * actuator];
    const char * jointName = mj_id2name(&model, mjOBJ_JOINT, joint);
    if(!jointName)
    {
      parity = false;
      continue;
    }
    const std::string name{jointName};
    const double limit = name.find("_steer") != std::string::npos ? 25.0 : 35.0;
    parity = parity && model.actuator_ctrllimited[actuator] && close(model.actuator_ctrlrange[2 * actuator], -limit)
             && close(model.actuator_ctrlrange[2 * actuator + 1], limit);
  }
  if(!parity)
  {
    std::cerr << "static parity diagnostic: nq=" << model.nq << " nv=" << model.nv << " njnt=" << model.njnt
              << " nu=" << model.nu << '\n';
    for(int joint = 0; joint < model.njnt; ++joint)
    {
      const char * name = mj_id2name(&model, mjOBJ_JOINT, joint);
      std::cerr << "  joint[" << joint << "]=" << (name ? name : "<unnamed>") << " type=" << model.jnt_type[joint]
                << " limited=" << static_cast<int>(model.jnt_limited[joint]) << '\n';
    }
    for(const auto & wheel : wheels)
    {
      std::cerr << "  expected wheel body=" << wheel.body << " drive=" << wheel.drive << " steer=" << wheel.steer
                << '\n';
    }
    const std::vector<std::string> bodies = {prefix + "chassis", prefix + "front_left_knuckle",
                                             prefix + "front_left_wheel"};
    for(const auto & bodyName : bodies)
    {
      const int body = mj_name2id(&model, mjOBJ_BODY, bodyName.c_str());
      if(body < 0) { continue; }
      std::cerr << "  body=" << bodyName << " mass=" << model.body_mass[body] << " ipos="
                << model.body_ipos[3 * body] << ',' << model.body_ipos[3 * body + 1] << ','
                << model.body_ipos[3 * body + 2] << " inertia=" << model.body_inertia[3 * body] << ','
                << model.body_inertia[3 * body + 1] << ',' << model.body_inertia[3 * body + 2] << '\n';
    }
    for(int actuator = 0; actuator < model.nu; ++actuator)
    {
      const int joint = model.actuator_trnid[2 * actuator];
      const char * name = mj_id2name(&model, mjOBJ_JOINT, joint);
      std::cerr << "  actuator=" << actuator << " joint=" << (name ? name : "<unnamed>") << " limited="
                << static_cast<int>(model.actuator_ctrllimited[actuator]) << " range="
                << model.actuator_ctrlrange[2 * actuator] << ',' << model.actuator_ctrlrange[2 * actuator + 1] << '\n';
    }
  }
  return parity;
}

// Every geom carried by the mc_mujoco ground body is terrain. The stock
// env/ground.xml has exactly one of them (`ground_floor`); the ramp terrain in
// src/mc_robots/rolling_contact_description/mujoco/ramp_terrain.xml adds twelve
// ramp slabs to the same body. Contact attribution, friction and the contact
// margin all have to cover the whole set: attributing contacts to the plane
// alone makes a wheel that has driven onto a ramp read as zero normal force,
// which the contact-mode estimator then reports as `detached`.
std::vector<int> collectTerrainGeoms(const mjModel & model)
{
  const int floor = namedId(model, mjOBJ_GEOM, "ground_floor");
  const int body = model.geom_bodyid[floor];
  std::vector<int> geoms;
  for(int geom = 0; geom < model.ngeom; ++geom)
  {
    if(model.geom_bodyid[geom] == body) { geoms.push_back(geom); }
  }
  return geoms;
}

bool isTerrain(const std::vector<int> & terrainGeoms, int geom)
{
  return std::find(terrainGeoms.begin(), terrainGeoms.end(), geom) != terrainGeoms.end();
}

void configureTerrain(mc_mujoco::MjSim & simulation,
                      const Options & options,
                      const std::string & robotName,
                      const std::vector<Wheel> & wheels,
                      Eigen::Vector3d & normal,
                      Eigen::Vector3d & tangentX,
                      Eigen::Vector3d & tangentY)
{
  auto & model = simulation.model();
  const int ground = namedId(model, mjOBJ_GEOM, "ground_floor");
  const double angle = options.rampDegrees * pi / 180.0;
  const Eigen::AngleAxisd orientation(-angle, Eigen::Vector3d::UnitY());
  const Eigen::Quaterniond quaternion{orientation};
  const int groundBody = model.geom_bodyid[ground];
  if(groundBody == 0)
  {
    throw std::runtime_error("the mc_mujoco ground plane is attached to world and cannot be rotated at runtime");
  }
  model.body_quat[4 * groundBody] = quaternion.w();
  model.body_quat[4 * groundBody + 1] = quaternion.x();
  model.body_quat[4 * groundBody + 2] = quaternion.y();
  model.body_quat[4 * groundBody + 3] = quaternion.z();
  for(const int geom : collectTerrainGeoms(model))
  {
    model.geom_friction[3 * geom] = options.frictionCycle >= 0 ? 0.8 : options.friction;
    model.geom_friction[3 * geom + 1] = 0.01;
    model.geom_friction[3 * geom + 2] = 0.001;
  }
  // World-geom transforms are part of MuJoCo's derived model constants.
  // Recompute them before resetting the robot onto the rotated plane.
  mj_setConst(&model, &simulation.data());

  tangentX = orientation * Eigen::Vector3d::UnitX();
  tangentY = Eigen::Vector3d::UnitY();
  normal = orientation * Eigen::Vector3d::UnitZ();
  // A 0.1 mm preload on a rotated plane avoids an ambiguous exactly tangent
  // cylinder/plane manifold in MuJoCo. Flat resets remain exactly tangent.
  const double contactPreload = options.rampDegrees == 0.0 ? 0.0 : 1e-4;
  const double chassisHeight = wheels.front().radius
                               + (options.robot == "differential" ? 0.0 : rangerChassisToAxle);
  // The lane offset is expressed in the terrain tangent frame so that it keeps
  // the chassis on the plane for a uniformly tilted world as well as for the
  // flat approach apron of the ramp terrain.
  const Eigen::Vector3d position =
      (chassisHeight - contactPreload) * normal + options.startX * tangentX + options.startY * tangentY;
  const sva::PTransformd pose(orientation.toRotationMatrix().transpose(), position);
  std::map<std::string, std::vector<double>> initialJoints;
  if(options.presetSteering && options.robot == "four-steering"
     && (options.scenario == "crab" || options.scenario == "ackermann_left"
         || options.scenario == "ackermann_right" || options.scenario == "pure_yaw"))
  {
    std::vector<double> q(8, 0.0);
    double linear = options.linearSpeed;
    double yaw = 0.0;
    if(options.scenario == "crab")
    {
      for(size_t i = 0; i < wheels.size(); ++i) { q[2 * i] = options.steeringAngle; }
    }
    else
    {
      if(options.scenario == "ackermann_left") { yaw = std::abs(options.yawRate); }
      else if(options.scenario == "ackermann_right") { yaw = -std::abs(options.yawRate); }
      else
      {
        linear = 0.0;
        yaw = options.yawRate;
      }
      for(size_t i = 0; i < wheels.size(); ++i)
      {
        q[2 * i] = std::atan2(yaw * wheels[i].offset.x(), linear - yaw * wheels[i].offset.y());
        if(q[2 * i] > 0.5 * pi) { q[2 * i] -= pi; }
        else if(q[2 * i] < -0.5 * pi) { q[2 * i] += pi; }
      }
    }
    initialJoints.emplace(robotName, std::move(q));
  }
  simulation.resetSimulation(initialJoints, {{robotName, pose}});
  auto & data = simulation.data();
  const int mocapId = model.body_mocapid[groundBody];
  if(mocapId >= 0)
  {
    data.mocap_quat[4 * mocapId] = quaternion.w();
    data.mocap_quat[4 * mocapId + 1] = quaternion.x();
    data.mocap_quat[4 * mocapId + 2] = quaternion.y();
    data.mocap_quat[4 * mocapId + 3] = quaternion.z();
  }
  else if(model.body_jntnum[groundBody] > 0)
  {
    const int groundJoint = model.body_jntadr[groundBody];
    if(model.jnt_type[groundJoint] != mjJNT_FREE)
    {
      throw std::runtime_error("the mc_mujoco ground body has an unsupported movable root joint");
    }
    const int groundQpos = model.jnt_qposadr[groundJoint];
    data.qpos[groundQpos + 3] = quaternion.w();
    data.qpos[groundQpos + 4] = quaternion.x();
    data.qpos[groundQpos + 5] = quaternion.y();
    data.qpos[groundQpos + 6] = quaternion.z();
  }
  else
  {
    // resetSimulation restores the initial pose of fixed environment robots
    // directly in mjModel. Reapply the ramp after that reset; unlike a free or
    // mocap body there is no mjData pose that can override it.
    model.body_quat[4 * groundBody] = quaternion.w();
    model.body_quat[4 * groundBody + 1] = quaternion.x();
    model.body_quat[4 * groundBody + 2] = quaternion.y();
    model.body_quat[4 * groundBody + 3] = quaternion.z();
  }
  mj_forward(&model, &data);
}

Eigen::Vector3d pointVelocity(const mjModel & model,
                              const mjData & data,
                              int body,
                              const Eigen::Vector3d & point,
                              std::vector<mjtNum> & jacobianPosition,
                              std::vector<mjtNum> & jacobianRotation)
{
  mj_jac(&model, &data, jacobianPosition.data(), jacobianRotation.data(), point.data(), body);
  Eigen::Vector3d out = Eigen::Vector3d::Zero();
  for(int row = 0; row < 3; ++row)
  {
    for(int columnIndex = 0; columnIndex < model.nv; ++columnIndex)
    {
      out[row] += jacobianPosition[static_cast<size_t>(row * model.nv + columnIndex)] * data.qvel[columnIndex];
    }
  }
  return out;
}

std::vector<WheelSample> sampleWheels(const mjModel & model,
                                      const mjData & data,
                                      const std::vector<Wheel> & wheels,
                                      const std::vector<int> & terrainGeoms,
                                      double friction,
                                      const Eigen::Vector3d & normal,
                                      std::vector<mjtNum> & jacobianPosition,
                                      std::vector<mjtNum> & jacobianRotation)
{
  std::vector<WheelSample> samples(wheels.size());
  for(size_t i = 0; i < wheels.size(); ++i)
  {
    const auto & wheel = wheels[i];
    auto & sample = samples[i];
    const int qpos = model.jnt_qposadr[wheel.driveJointId];
    const int qvel = model.jnt_dofadr[wheel.driveJointId];
    sample.q = data.qpos[qpos];
    sample.qd = data.qvel[qvel];
    if(wheel.steerJointId >= 0)
    {
      sample.steering = data.qpos[model.jnt_qposadr[wheel.steerJointId]];
      sample.steerTorque = data.qfrc_actuator[model.jnt_dofadr[wheel.steerJointId]];
    }

    const Eigen::Vector3d center{data.xpos[3 * wheel.bodyId], data.xpos[3 * wheel.bodyId + 1],
                                 data.xpos[3 * wheel.bodyId + 2]};
    const Eigen::Vector3d point = center - wheel.radius * normal;
    Eigen::Vector3d lateral = column(&data.xmat[9 * wheel.bodyId], 1);
    lateral -= normal * lateral.dot(normal);
    lateral.normalize();
    const Eigen::Vector3d rolling = lateral.cross(normal).normalized();
    mjtNum bodyVelocity[6] = {};
    mj_objectVelocity(&model, &data, mjOBJ_BODY, wheel.bodyId, bodyVelocity, 0);
    sample.totalSpinRate =
        Eigen::Vector3d{bodyVelocity[0], bodyVelocity[1], bodyVelocity[2]}.dot(lateral);
    const Eigen::Vector3d velocity =
        pointVelocity(model, data, wheel.bodyId, point, jacobianPosition, jacobianRotation);
    sample.rollingSlip = velocity.dot(rolling);
    sample.lateralSlip = velocity.dot(lateral);
    sample.normalSpeed = velocity.dot(normal);
    sample.driveTorque = data.qfrc_actuator[qvel];
  }

  for(int contactIndex = 0; contactIndex < data.ncon; ++contactIndex)
  {
    const auto & contact = data.contact[contactIndex];
    const int wheelGeom = isTerrain(terrainGeoms, contact.geom[0])
                              ? contact.geom[1]
                              : (isTerrain(terrainGeoms, contact.geom[1]) ? contact.geom[0] : -1);
    if(wheelGeom < 0) { continue; }
    const auto wheel = std::find_if(wheels.begin(), wheels.end(),
                                    [wheelGeom](const Wheel & candidate) { return candidate.geomId == wheelGeom; });
    if(wheel == wheels.end()) { continue; }
    const size_t index = static_cast<size_t>(std::distance(wheels.begin(), wheel));
    mjtNum force[6] = {};
    mj_contactForce(&model, &data, contactIndex, force);
    samples[index].normalForce += std::max(0.0, static_cast<double>(force[0]));
    samples[index].tangentialForce += std::hypot(static_cast<double>(force[1]), static_cast<double>(force[2]));
    // Unsigned tilt of the real contact plane against the assumed one; the
    // absolute value makes it independent of which of the pair MuJoCo made
    // geom[0], since a plane and its opposite normal are the same plane.
    const Eigen::Vector3d contactNormal{contact.frame[0], contact.frame[1], contact.frame[2]};
    const double alignment = std::min(1.0, std::abs(contactNormal.normalized().dot(normal)));
    samples[index].contactNormalDeviation = std::max(samples[index].contactNormalDeviation, std::acos(alignment));
    ++samples[index].contacts;
  }
  for(auto & sample : samples)
  {
    sample.frictionMargin = friction * sample.normalForce - sample.tangentialForce;
  }
  return samples;
}

void sendMeasurements(mc_mujoco::MjSim & simulation,
                      const std::vector<Wheel> & wheels,
                      const std::vector<WheelSample> & samples)
{
  auto * global = simulation.controller();
  if(!global) { return; }
  auto & store = global->controller().datastore();
  constexpr const char * callback = "RollingContact::SetMeasuredContact";
  if(!store.has(callback)) { return; }
  for(size_t i = 0; i < wheels.size(); ++i)
  {
    const std::string & name = wheels[i].name;
    store.call<void, const std::string &, double, double, double, double, bool>(
        callback, name, samples[i].rollingSlip, samples[i].lateralSlip, samples[i].normalForce,
        samples[i].tangentialForce, true);
  }
}

bool readControllerState(mc_mujoco::MjSim & simulation,
                         const std::vector<Wheel> & wheels,
                         std::vector<WheelSample> & samples)
{
  auto * global = simulation.controller();
  if(!global) { return false; }
  auto & store = global->controller().datastore();
  constexpr const char * modeCallback = "RollingContact::GetEstimatedMode";
  constexpr const char * activationCallback = "RollingContact::GetActivation";
  constexpr const char * solverActivationCallback = "RollingContact::GetSolverActivation";
  constexpr const char * residualCallback = "RollingContact::GetAccelerationResidual";
  constexpr const char * lateralAccelerationCallback = "RollingContact::GetLateralAccelerationResidual";
  constexpr const char * normalAccelerationCallback = "RollingContact::GetNormalAccelerationResidual";
  constexpr const char * hardPromotionResidualCallback = "RollingContact::GetHardPromotionResidual";
  constexpr const char * qpNormalForceCallback = "RollingContact::GetQPNormalForce";
  constexpr const char * qpTangentialForceCallback = "RollingContact::GetQPTangentialForce";
  constexpr const char * generatorCallback = "RollingContact::GetSlidingGenerator";
  constexpr const char * targetCallback = "RollingContact::GetDriveTarget";
  constexpr const char * accelerationCallback = "RollingContact::GetDriveAcceleration";
  constexpr const char * positionCallback = "RollingContact::GetDrivePosition";
  constexpr const char * velocityCallback = "RollingContact::GetDriveVelocity";
  constexpr const char * fallbackCallback = "RollingContact::GetContactFallback";
  if(!store.has(modeCallback) || !store.has(activationCallback) || !store.has(solverActivationCallback)
     || !store.has(residualCallback) || !store.has(lateralAccelerationCallback)
     || !store.has(normalAccelerationCallback) || !store.has(hardPromotionResidualCallback)
     || !store.has(qpNormalForceCallback) || !store.has(qpTangentialForceCallback)
     || !store.has(generatorCallback)
     || !store.has(targetCallback) || !store.has(accelerationCallback) || !store.has(positionCallback)
     || !store.has(velocityCallback) || !store.has(fallbackCallback))
  {
    throw std::runtime_error("rolling-contact state callbacks are unavailable");
  }
  for(size_t i = 0; i < wheels.size(); ++i)
  {
    const std::string & name = wheels[i].name;
    samples[i].mode = store.call<std::string, const std::string &>(modeCallback, name);
    samples[i].activation = store.call<double, const std::string &>(activationCallback, name);
    samples[i].solverActivation = store.call<double, const std::string &>(solverActivationCallback, name);
    samples[i].accelerationResidual = store.call<double, const std::string &>(residualCallback, name);
    samples[i].lateralAccelerationResidual =
        store.call<double, const std::string &>(lateralAccelerationCallback, name);
    samples[i].normalAccelerationResidual =
        store.call<double, const std::string &>(normalAccelerationCallback, name);
    samples[i].hardPromotionResidual =
        store.call<double, const std::string &>(hardPromotionResidualCallback, name);
    samples[i].qpNormalForce = store.call<double, const std::string &>(qpNormalForceCallback, name);
    samples[i].qpTangentialForce = store.call<double, const std::string &>(qpTangentialForceCallback, name);
    samples[i].slidingGenerator = store.call<int, const std::string &>(generatorCallback, name);
    samples[i].driveTarget = store.call<double, const std::string &>(targetCallback, name);
    samples[i].driveAcceleration = store.call<double, const std::string &>(accelerationCallback, name);
    samples[i].controllerDrivePosition = store.call<double, const std::string &>(positionCallback, name);
    samples[i].controllerDriveVelocity = store.call<double, const std::string &>(velocityCallback, name);
  }
  return store.call<bool>(fallbackCallback);
}

ControllerReference readControllerReference(mc_mujoco::MjSim & simulation)
{
  auto * global = simulation.controller();
  if(!global) { throw std::runtime_error("mc_mujoco controller is unavailable"); }
  auto & store = global->controller().datastore();
  constexpr const char * positionCallback = "RollingContact::GetBasePositionTarget";
  constexpr const char * yawCallback = "RollingContact::GetBaseYawTarget";
  constexpr const char * postureEvalCallback = "RollingContact::GetPostureEvalNorm";
  constexpr const char * positionEvalCallback = "RollingContact::GetPositionEvalNorm";
  constexpr const char * orientationEvalCallback = "RollingContact::GetOrientationEvalNorm";
  constexpr const char * hardRhsCallback = "RollingContact::GetHardRhsNorm";
  constexpr const char * rollingWeightCallback = "RollingContact::GetRollingWeight";
  constexpr const char * dynamicsResidualCallback = "RollingContact::GetDynamicsResidual";
  constexpr const char * floatingBaseEffortCallback = "RollingContact::GetFloatingBaseEffortNorm";
  if(!store.has(positionCallback) || !store.has(yawCallback) || !store.has(postureEvalCallback)
     || !store.has(positionEvalCallback) || !store.has(orientationEvalCallback) || !store.has(hardRhsCallback)
     || !store.has(rollingWeightCallback) || !store.has(dynamicsResidualCallback)
     || !store.has(floatingBaseEffortCallback))
  {
    throw std::runtime_error("rolling-contact reference callbacks are unavailable");
  }
  return {store.call<Eigen::Vector3d>(positionCallback), store.call<double>(yawCallback),
          store.call<double>(postureEvalCallback), store.call<double>(positionEvalCallback),
          store.call<double>(orientationEvalCallback), store.call<double>(hardRhsCallback),
          store.call<double>(rollingWeightCallback), store.call<double>(dynamicsResidualCallback),
          store.call<double>(floatingBaseEffortCallback)};
}

void requireControllerBackend(mc_mujoco::MjSim & simulation, const std::string & expected)
{
  auto * global = simulation.controller();
  if(!global) { throw std::runtime_error("mc_mujoco controller is unavailable"); }
  auto & store = global->controller().datastore();
  constexpr const char * callback = "RollingContact::GetBackend";
  if(!store.has(callback)) { throw std::runtime_error("rolling-contact backend callback is unavailable"); }
  const std::string actual = store.call<std::string>(callback);
  if(actual != expected)
  {
    throw std::runtime_error("configured controller backend is " + actual + ", expected " + expected);
  }
}

Eigen::Vector3d reconstructedTwist(const Options & options,
                                   const std::vector<Wheel> & wheels,
                                   const std::vector<WheelSample> & samples)
{
  if(options.robot == "differential")
  {
    const double left = wheels[0].radius * samples[0].totalSpinRate;
    const double right = wheels[1].radius * samples[1].totalSpinRate;
    return {(left + right) * 0.5, 0.0, (right - left) / 0.6};
  }
  Eigen::MatrixXd matrix(static_cast<Eigen::Index>(2 * wheels.size()), 3);
  Eigen::VectorXd rhs(static_cast<Eigen::Index>(2 * wheels.size()));
  for(size_t i = 0; i < wheels.size(); ++i)
  {
    const double c = std::cos(samples[i].steering);
    const double s = std::sin(samples[i].steering);
    const double x = wheels[i].offset.x();
    const double y = wheels[i].offset.y();
    matrix.row(static_cast<Eigen::Index>(2 * i)) << c, s, -y * c + x * s;
    matrix.row(static_cast<Eigen::Index>(2 * i + 1)) << -s, c, y * s + x * c;
    rhs[static_cast<Eigen::Index>(2 * i)] = wheels[i].radius * samples[i].totalSpinRate;
    rhs[static_cast<Eigen::Index>(2 * i + 1)] = 0.0;
  }
  return matrix.colPivHouseholderQr().solve(rhs);
}

bool finiteState(const mjModel & model, const mjData & data)
{
  for(int i = 0; i < model.nq; ++i)
  {
    if(!std::isfinite(data.qpos[i])) { return false; }
  }
  for(int i = 0; i < model.nv; ++i)
  {
    if(!std::isfinite(data.qvel[i])) { return false; }
  }
  for(int i = 0; i < model.nu; ++i)
  {
    if(!std::isfinite(data.ctrl[i])) { return false; }
  }
  return true;
}

void updateStatistics(Statistics & statistics,
                      const std::vector<WheelSample> & samples,
                      bool contactFallback,
                      double trajectoryError,
                      double yawError,
                      double trackingPositionError,
                      double trackingYawError,
                      double cycleWallMilliseconds)
{
  ++statistics.samples;
  if(contactFallback) { ++statistics.contactFallbackSamples; }
  bool anyMissing = false;
  for(const auto & sample : samples)
  {
    statistics.maxRollingSlip = std::max(statistics.maxRollingSlip, std::abs(sample.rollingSlip));
    statistics.maxLateralSlip = std::max(statistics.maxLateralSlip, std::abs(sample.lateralSlip));
    statistics.maxNormalSpeed = std::max(statistics.maxNormalSpeed, std::abs(sample.normalSpeed));
    statistics.minNormalForce = std::min(statistics.minNormalForce, sample.normalForce);
    statistics.minFrictionMargin = std::min(statistics.minFrictionMargin, sample.frictionMargin);
    statistics.maxDriveTorque = std::max(statistics.maxDriveTorque, std::abs(sample.driveTorque));
    statistics.maxSteerTorque = std::max(statistics.maxSteerTorque, std::abs(sample.steerTorque));
    statistics.maxContactNormalDeviation =
        std::max(statistics.maxContactNormalDeviation, sample.contactNormalDeviation);
    statistics.rollingSlips.push_back(std::abs(sample.rollingSlip));
    statistics.lateralSlips.push_back(std::abs(sample.lateralSlip));
    statistics.normalSpeeds.push_back(std::abs(sample.normalSpeed));
    statistics.modes.insert(sample.mode);
    anyMissing = anyMissing || sample.contacts == 0;
  }
  if(anyMissing) { ++statistics.noContactSamples; }
  statistics.maxTrajectoryError = std::max(statistics.maxTrajectoryError, trajectoryError);
  statistics.trajectoryErrorSquared += trajectoryError * trajectoryError;
  statistics.maxYawError = std::max(statistics.maxYawError, std::abs(yawError));
  statistics.maxTrackingPositionError = std::max(statistics.maxTrackingPositionError, trackingPositionError);
  statistics.trackingPositionErrorSquared += trackingPositionError * trackingPositionError;
  statistics.maxTrackingYawError = std::max(statistics.maxTrackingYawError, std::abs(trackingYawError));
  statistics.trackingYawErrorSquared += trackingYawError * trackingYawError;
  statistics.odometryPositionErrors.push_back(trajectoryError);
  statistics.odometryYawErrors.push_back(std::abs(yawError));
  statistics.trackingPositionErrors.push_back(trackingPositionError);
  statistics.trackingYawErrors.push_back(std::abs(trackingYawError));
  statistics.cycleWallMilliseconds.push_back(cycleWallMilliseconds);
  if(cycleWallMilliseconds >= 5.0) { ++statistics.missedDeadlines; }
}

void writeCsvHeader(std::ofstream & stream, const std::vector<Wheel> & wheels)
{
  stream << "cycle,time,base_x,base_y,base_z,terrain_x,terrain_y,yaw,odometry_x,odometry_y,odometry_yaw,"
            "odometry_position_error,odometry_yaw_error,target_x,target_y,target_yaw,tracking_position_error,"
            "tracking_yaw_error,posture_eval_norm,position_eval_norm,orientation_eval_norm,hard_rhs_norm,"
            "rolling_weight,dynamics_residual,floating_base_effort_norm,cycle_wall_ms,contacts,contact_fallback";
  for(const auto & wheel : wheels)
  {
    stream << ',' << wheel.name << "_q," << wheel.name << "_qd," << wheel.name << "_total_spin_rate," << wheel.name
           << "_steering," << wheel.name
           << "_rolling_slip," << wheel.name << "_lateral_slip," << wheel.name << "_normal_speed," << wheel.name
           << "_normal_force," << wheel.name << "_tangential_force," << wheel.name << "_friction_margin," << wheel.name
           << "_drive_torque," << wheel.name << "_steer_torque," << wheel.name << "_drive_target," << wheel.name
           << "_drive_acceleration," << wheel.name << "_controller_q," << wheel.name << "_controller_qd," << wheel.name
           << "_contacts";
    stream << ',' << wheel.name << "_mode," << wheel.name << "_activation," << wheel.name << "_solver_activation,"
           << wheel.name << "_acceleration_residual," << wheel.name << "_lateral_acceleration_residual," << wheel.name
           << "_normal_acceleration_residual," << wheel.name << "_hard_promotion_residual," << wheel.name
           << "_qp_normal_force," << wheel.name << "_qp_tangential_force," << wheel.name << "_sliding_generator,"
           << wheel.name << "_contact_normal_deviation";
  }
  stream << '\n';
}

void writeCsvRow(std::ofstream & stream,
                 size_t cycle,
                 const mjData & data,
                 const Eigen::Vector3d & base,
                 const Eigen::Vector2d & terrainPosition,
                 double yaw,
                 const Eigen::Vector3d & odometry,
                 double trajectoryError,
                 double yawError,
                 const Eigen::Vector2d & targetPosition,
                 double targetYaw,
                 double trackingPositionError,
                 double trackingYawError,
                 const ControllerReference & reference,
                 double cycleWallMilliseconds,
                 bool contactFallback,
                 const std::vector<WheelSample> & samples)
{
  int contacts = 0;
  for(const auto & sample : samples) { contacts += sample.contacts; }
  stream << cycle << ',' << data.time << ',' << base.x() << ',' << base.y() << ',' << base.z() << ','
         << terrainPosition.x() << ',' << terrainPosition.y() << ',' << yaw << ',' << odometry.x() << ','
         << odometry.y() << ',' << odometry.z() << ',' << trajectoryError << ',' << yawError << ','
         << targetPosition.x() << ',' << targetPosition.y() << ',' << targetYaw << ',' << trackingPositionError << ','
         << trackingYawError << ',' << reference.postureEvalNorm << ',' << reference.positionEvalNorm << ','
         << reference.orientationEvalNorm << ',' << reference.hardRhsNorm << ',' << reference.rollingWeight << ','
         << reference.dynamicsResidual << ',' << reference.floatingBaseEffortNorm << ',' << cycleWallMilliseconds << ','
         << contacts << ',' << (contactFallback ? 1 : 0);
  for(const auto & sample : samples)
  {
    stream << ',' << sample.q << ',' << sample.qd << ',' << sample.totalSpinRate << ',' << sample.steering << ','
           << sample.rollingSlip << ',' << sample.lateralSlip << ',' << sample.normalSpeed << ',' << sample.normalForce
           << ','
           << sample.tangentialForce << ',' << sample.frictionMargin << ',' << sample.driveTorque << ','
           << sample.steerTorque << ',' << sample.driveTarget << ',' << sample.driveAcceleration << ','
           << sample.controllerDrivePosition << ',' << sample.controllerDriveVelocity << ',' << sample.contacts << ','
           << sample.mode << ',' << sample.activation << ',' << sample.solverActivation << ','
           << sample.accelerationResidual << ',' << sample.lateralAccelerationResidual << ','
           << sample.normalAccelerationResidual << ',' << sample.hardPromotionResidual << ',' << sample.qpNormalForce
           << ',' << sample.qpTangentialForce << ',' << sample.slidingGenerator << ','
           << sample.contactNormalDeviation;
  }
  stream << '\n';
}

void writeReport(const Options & options,
                 const Statistics & statistics,
                 size_t completedCycles,
                 bool staticParity,
                 bool controllerFailed,
                 bool finite,
                 double simulationTime,
                 double wallSeconds,
                 double cpuSeconds,
                 long peakResidentKiB,
                 const Eigen::Vector3d & finalOdometry,
                 const Eigen::Vector2d & finalPosition,
                 double finalYaw,
                 bool finalContactFallback,
                 const std::vector<WheelSample> & finalSamples)
{
  std::ofstream stream(options.report);
  if(!stream) { throw std::runtime_error("cannot open report: " + options.report); }
  const double odometryRms = statistics.samples == 0
                                 ? 0.0
                                 : std::sqrt(statistics.trajectoryErrorSquared
                                             / static_cast<double>(statistics.samples));
  const double trackingPositionRms = statistics.samples == 0
                                         ? 0.0
                                         : std::sqrt(statistics.trackingPositionErrorSquared
                                                     / static_cast<double>(statistics.samples));
  const double trackingYawRms = statistics.samples == 0
                                    ? 0.0
                                    : std::sqrt(statistics.trackingYawErrorSquared
                                                / static_cast<double>(statistics.samples));
  const double minimumNormalForce = statistics.samples == 0 ? 0.0 : statistics.minNormalForce;
  const double minimumFrictionMargin = statistics.samples == 0 ? 0.0 : statistics.minFrictionMargin;
  const double noContactFraction = statistics.samples == 0
                                       ? 1.0
                                       : static_cast<double>(statistics.noContactSamples)
                                             / static_cast<double>(statistics.samples);
  const double contactFallbackFraction = statistics.samples == 0
                                             ? 0.0
                                             : static_cast<double>(statistics.contactFallbackSamples)
                                                   / static_cast<double>(statistics.samples);
  stream << std::setprecision(17);
  stream << "{\n"
         // 4 adds start_x_m, start_y_m and max_contact_normal_deviation_deg.
         << "  \"schema\": 4,\n"
         << "  \"device\": \"CPU\",\n"
         << "  \"robot\": \"" << options.robot << "\",\n"
         << "  \"scenario\": \"" << options.scenario << "\",\n"
         << "  \"backend\": \"" << options.backend << "\",\n"
         // Which regime this replay actually ran in. Without these a
         // report cannot be compared against a `mc_mujoco -f <config>` run.
         << "  \"torque_control\": " << (options.torqueControl ? "true" : "false") << ",\n"
         << "  \"measured_contacts\": " << (options.measuredContacts ? "true" : "false") << ",\n"
         << "  \"preset_steering\": " << (options.presetSteering ? "true" : "false") << ",\n"
         << "  \"cycles_requested\": " << options.cycles << ",\n"
         << "  \"cycles_completed\": " << completedCycles << ",\n"
         << "  \"warmup_cycles\": " << options.warmupCycles << ",\n"
         << "  \"measured_samples\": " << statistics.samples << ",\n"
         << "  \"static_parity\": " << (staticParity ? "true" : "false") << ",\n"
         << "  \"controller_failed\": " << (controllerFailed ? "true" : "false") << ",\n"
         << "  \"finite_state\": " << (finite ? "true" : "false") << ",\n"
         << "  \"ramp_degrees\": " << options.rampDegrees << ",\n"
         << "  \"start_x_m\": " << options.startX << ",\n"
         << "  \"start_y_m\": " << options.startY << ",\n"
         << "  \"friction\": " << options.friction << ",\n"
         << "  \"linear_speed_mps\": " << options.linearSpeed << ",\n"
         << "  \"yaw_rate_radps\": " << options.yawRate << ",\n"
         << "  \"steering_angle_rad\": " << options.steeringAngle << ",\n"
         << "  \"impulse_direction\": \"" << options.impulseDirection << "\",\n"
         << "  \"simulation_time_s\": " << simulationTime << ",\n"
         << "  \"wall_time_s\": " << wallSeconds << ",\n"
         << "  \"cpu_time_s\": " << cpuSeconds << ",\n"
         << "  \"peak_resident_kib\": " << peakResidentKiB << ",\n"
         << "  \"real_time_factor\": " << (wallSeconds > 0.0 ? simulationTime / wallSeconds : 0.0) << ",\n"
         << "  \"max_rolling_slip_mps\": " << statistics.maxRollingSlip << ",\n"
         << "  \"rolling_slip_rms_mps\": " << rms(statistics.rollingSlips) << ",\n"
         << "  \"rolling_slip_p95_mps\": " << percentile(statistics.rollingSlips, 0.95) << ",\n"
         << "  \"max_lateral_slip_mps\": " << statistics.maxLateralSlip << ",\n"
         << "  \"lateral_slip_rms_mps\": " << rms(statistics.lateralSlips) << ",\n"
         << "  \"lateral_slip_p95_mps\": " << percentile(statistics.lateralSlips, 0.95) << ",\n"
         << "  \"max_normal_speed_mps\": " << statistics.maxNormalSpeed << ",\n"
         << "  \"normal_speed_rms_mps\": " << rms(statistics.normalSpeeds) << ",\n"
         << "  \"normal_speed_p95_mps\": " << percentile(statistics.normalSpeeds, 0.95) << ",\n"
         << "  \"min_normal_force_n\": " << minimumNormalForce << ",\n"
         << "  \"min_friction_margin_n\": " << minimumFrictionMargin << ",\n"
         << "  \"max_drive_torque_nm\": " << statistics.maxDriveTorque << ",\n"
         << "  \"max_steer_torque_nm\": " << statistics.maxSteerTorque << ",\n"
         << "  \"max_contact_normal_deviation_deg\": "
         << statistics.maxContactNormalDeviation * 180.0 / pi << ",\n"
         << "  \"no_contact_sample_fraction\": " << noContactFraction << ",\n"
         << "  \"contact_fallback_sample_fraction\": " << contactFallbackFraction << ",\n"
         << "  \"odometry_position_error_rms_m\": " << odometryRms << ",\n"
         << "  \"odometry_position_error_p95_m\": "
         << percentile(statistics.odometryPositionErrors, 0.95) << ",\n"
         << "  \"odometry_position_error_max_m\": " << statistics.maxTrajectoryError << ",\n"
         << "  \"odometry_yaw_error_rms_rad\": " << rms(statistics.odometryYawErrors) << ",\n"
         << "  \"odometry_yaw_error_p95_rad\": " << percentile(statistics.odometryYawErrors, 0.95) << ",\n"
         << "  \"odometry_yaw_error_max_rad\": " << statistics.maxYawError << ",\n"
         << "  \"tracking_position_error_rms_m\": " << trackingPositionRms << ",\n"
         << "  \"tracking_position_error_p95_m\": " << percentile(statistics.trackingPositionErrors, 0.95) << ",\n"
         << "  \"tracking_position_error_max_m\": " << statistics.maxTrackingPositionError << ",\n"
         << "  \"tracking_yaw_error_rms_rad\": " << trackingYawRms << ",\n"
         << "  \"tracking_yaw_error_p95_rad\": " << percentile(statistics.trackingYawErrors, 0.95) << ",\n"
         << "  \"tracking_yaw_error_max_rad\": " << statistics.maxTrackingYawError << ",\n"
         << "  \"cycle_wall_median_ms\": " << percentile(statistics.cycleWallMilliseconds, 0.5) << ",\n"
         << "  \"cycle_wall_p95_ms\": " << percentile(statistics.cycleWallMilliseconds, 0.95) << ",\n"
         << "  \"cycle_wall_p99_ms\": " << percentile(statistics.cycleWallMilliseconds, 0.99) << ",\n"
         << "  \"cycle_wall_max_ms\": "
         << (statistics.cycleWallMilliseconds.empty()
                 ? 0.0
                 : *std::max_element(statistics.cycleWallMilliseconds.begin(), statistics.cycleWallMilliseconds.end()))
         << ",\n"
         << "  \"missed_deadlines\": " << statistics.missedDeadlines << ",\n"
         << "  \"estimated_modes\": [";
  bool firstMode = true;
  for(const auto & mode : statistics.modes)
  {
    if(!firstMode) { stream << ", "; }
    stream << '\"' << mode << '\"';
    firstMode = false;
  }
  stream << "],\n"
         << "  \"final_contact_fallback\": " << (finalContactFallback ? "true" : "false") << ",\n"
         << "  \"final_modes\": [";
  for(size_t i = 0; i < finalSamples.size(); ++i)
  {
    if(i != 0) { stream << ", "; }
    stream << '\"' << finalSamples[i].mode << '\"';
  }
  stream << "],\n"
         << "  \"final_solver_activations\": [";
  for(size_t i = 0; i < finalSamples.size(); ++i)
  {
    if(i != 0) { stream << ", "; }
    stream << finalSamples[i].solverActivation;
  }
  stream << "],\n"
         << "  \"final_position\": [" << finalPosition.x() << ", " << finalPosition.y() << "],\n"
         << "  \"final_yaw_rad\": " << finalYaw << ",\n"
         << "  \"final_odometry\": [" << finalOdometry.x() << ", " << finalOdometry.y() << ", "
         << finalOdometry.z() << "]\n"
         << "}\n";
}

} // namespace

int main(int argc, char ** argv)
{
  try
  {
    const Options options = parseOptions(argc, argv);
    mc_mujoco::MjConfiguration configuration;
    configuration.mc_config = options.mcConfig;
    configuration.with_visualization = false;
    configuration.with_mc_rtc_gui = false;
    configuration.with_controller = true;
    configuration.sync_real_time = false;
    configuration.step_by_step = false;
    configuration.torque_control = options.torqueControl;

    mc_mujoco::MjSim simulation(configuration);
    auto & model = simulation.model();
    auto & data = simulation.data();
    const std::string robotName = options.robot == "differential" ? "rolling_diff" : "ranger_mini_v3";
    std::vector<Wheel> wheels = makeWheels(options);
    for(auto & wheel : wheels)
    {
      wheel.body = robotName + "_" + wheel.body;
      wheel.drive = robotName + "_" + wheel.drive;
      if(!wheel.steer.empty()) { wheel.steer = robotName + "_" + wheel.steer; }
      wheel.bodyId = namedId(model, mjOBJ_BODY, wheel.body);
      wheel.geomId = namedId(model, mjOBJ_GEOM, robotName + "_" + wheel.name + "_wheel_collision");
      wheel.driveJointId = namedId(model, mjOBJ_JOINT, wheel.drive);
      if(!wheel.steer.empty()) { wheel.steerJointId = namedId(model, mjOBJ_JOINT, wheel.steer); }
    }
    const bool staticParity = checkStaticParity(model, options, wheels);
    if(!staticParity) { throw std::runtime_error("MuJoCo/RBDyn static-parity check failed"); }

    Eigen::Vector3d normal;
    Eigen::Vector3d tangentX;
    Eigen::Vector3d tangentY;
    configureTerrain(simulation, options, robotName, wheels, normal, tangentX, tangentY);
    if(options.frictionCycle < 0)
    {
      for(const auto & wheel : wheels) { model.geom_friction[3 * wheel.geomId] = options.friction; }
    }
    requireControllerBackend(simulation, options.backend);
    const int groundGeom = namedId(model, mjOBJ_GEOM, "ground_floor");
    const std::vector<int> terrainGeoms = collectTerrainGeoms(model);
    // Preserve the exact zero-gap reset while giving MuJoCo a small contact
    // discovery envelope. Without it, round-off on a coplanar four-wheel
    // reset can select only the front or rear axle for the first substep.
    for(const int geom : terrainGeoms)
    {
      model.geom_margin[geom] = std::max(model.geom_margin[geom], 1e-4);
    }
    for(const auto & wheel : wheels)
    {
      model.geom_margin[wheel.geomId] = std::max(model.geom_margin[wheel.geomId], 1e-4);
    }
    mj_forward(&model, &data);
    const Eigen::Vector3d simulatedGroundNormal = column(&data.geom_xmat[9 * groundGeom], 2);
    if((simulatedGroundNormal - normal).norm() > 1e-12)
    {
      throw std::runtime_error(
          "MuJoCo ground normal does not match the configured ramp normal: measured ["
          + std::to_string(simulatedGroundNormal.x()) + ", " + std::to_string(simulatedGroundNormal.y()) + ", "
          + std::to_string(simulatedGroundNormal.z()) + "], expected [" + std::to_string(normal.x()) + ", "
          + std::to_string(normal.y()) + ", " + std::to_string(normal.z()) + "]");
    }
    const Eigen::Vector3d groundPosition{data.geom_xpos[3 * groundGeom], data.geom_xpos[3 * groundGeom + 1],
                                         data.geom_xpos[3 * groundGeom + 2]};
    double maximumInitialWheelGap = 0.0;
    const double expectedWheelDistance = wheels.front().radius - (options.rampDegrees == 0.0 ? 0.0 : 1e-4);
    for(const auto & wheel : wheels)
    {
      const Eigen::Vector3d center{data.xpos[3 * wheel.bodyId], data.xpos[3 * wheel.bodyId + 1],
                                   data.xpos[3 * wheel.bodyId + 2]};
      maximumInitialWheelGap =
          std::max(maximumInitialWheelGap, std::abs(normal.dot(center - groundPosition) - expectedWheelDistance));
    }
    if(maximumInitialWheelGap > 1e-9)
    {
      throw std::runtime_error("ramp reset does not place every wheel on the terrain plane; maximum gap error is "
                               + std::to_string(maximumInitialWheelGap));
    }
    const int chassisBody = namedId(model, mjOBJ_BODY, robotName + "_chassis");
    const int rootJoint = namedId(model, mjOBJ_JOINT, robotName + "_root");
    const int rootVelocity = model.jnt_dofadr[rootJoint];
    const int rootPosition = model.jnt_qposadr[rootJoint];
    const size_t frameskip = static_cast<size_t>(std::llround(0.005 / model.opt.timestep));
    if(frameskip == 0 || !close(static_cast<double>(frameskip) * model.opt.timestep, 0.005))
    {
      throw std::runtime_error("MuJoCo timestep is not an integer divisor of the controller timestep");
    }

    const Eigen::Vector3d initialBase{data.xpos[3 * chassisBody], data.xpos[3 * chassisBody + 1],
                                      data.xpos[3 * chassisBody + 2]};
    const double initialYaw = yawInTerrain(data, chassisBody, tangentX, tangentY);
    for(auto & wheel : wheels) { wheel.previousQ = data.qpos[model.jnt_qposadr[wheel.driveJointId]]; }

    std::ofstream csv(options.csv);
    if(!csv) { throw std::runtime_error("cannot open CSV: " + options.csv); }
    csv << std::setprecision(17);
    writeCsvHeader(csv, wheels);

    Statistics statistics;
    Eigen::Vector3d odometry = Eigen::Vector3d::Zero();
    Eigen::Vector2d finalPosition = Eigen::Vector2d::Zero();
    double finalYaw = 0.0;
    bool controllerFailed = false;
    bool allFinite = true;
    bool finalContactFallback = false;
    size_t completedCycles = 0;
    std::vector<WheelSample> finalSamples(wheels.size());
    std::vector<mjtNum> jacobianPosition(static_cast<size_t>(3 * model.nv), 0.0);
    std::vector<mjtNum> jacobianRotation(static_cast<size_t>(3 * model.nv), 0.0);

    const auto wallStart = std::chrono::steady_clock::now();
    const std::clock_t cpuStart = std::clock();
    for(size_t cycle = 0; cycle < options.cycles; ++cycle)
    {
      const auto cycleStart = std::chrono::steady_clock::now();
      if(static_cast<long>(cycle) == options.frictionCycle)
      {
        for(const int geom : terrainGeoms) { model.geom_friction[3 * geom] = options.friction; }
        for(const auto & wheel : wheels) { model.geom_friction[3 * wheel.geomId] = options.friction; }
      }
      if(static_cast<long>(cycle) == options.impulseCycle)
      {
        const Eigen::Vector3d direction = options.impulseDirection == "longitudinal"
                                              ? tangentX
                                              : (options.impulseDirection == "lateral" ? tangentY : normal);
        data.qvel[rootVelocity] += options.impulse * direction.x();
        data.qvel[rootVelocity + 1] += options.impulse * direction.y();
        data.qvel[rootVelocity + 2] += options.impulse * direction.z();
      }
      if(static_cast<long>(cycle) == options.separationCycle)
      {
        data.qpos[rootPosition] += options.separation * normal.x();
        data.qpos[rootPosition + 1] += options.separation * normal.y();
        data.qpos[rootPosition + 2] += options.separation * normal.z();
        mj_forward(&model, &data);
      }

      for(size_t step = 0; step < frameskip; ++step)
      {
        controllerFailed = simulation.stepSimulation();
        if(options.measuredContacts)
        {
          const auto substepSamples = sampleWheels(model, data, wheels, terrainGeoms, options.friction, normal,
                                                   jacobianPosition, jacobianRotation);
          sendMeasurements(simulation, wheels, substepSamples);
        }
        allFinite = allFinite && finiteState(model, data);
        if(controllerFailed || !allFinite) { break; }
      }
      if(controllerFailed || !allFinite) { break; }

      const auto samples =
          sampleWheels(model, data, wheels, terrainGeoms, options.friction, normal, jacobianPosition,
                       jacobianRotation);
      auto samplesWithModes = samples;
      const bool contactFallback = readControllerState(simulation, wheels, samplesWithModes);
      finalContactFallback = contactFallback;
      finalSamples = samplesWithModes;
      const Eigen::Vector3d twist = reconstructedTwist(options, wheels, samplesWithModes);
      const double yawMidpoint = odometry.z() + 0.5 * 0.005 * twist.z();
      odometry.x() += 0.005 * (std::cos(yawMidpoint) * twist.x() - std::sin(yawMidpoint) * twist.y());
      odometry.y() += 0.005 * (std::sin(yawMidpoint) * twist.x() + std::cos(yawMidpoint) * twist.y());
      odometry.z() = wrapAngle(odometry.z() + 0.005 * twist.z());

      const Eigen::Vector3d base{data.xpos[3 * chassisBody], data.xpos[3 * chassisBody + 1],
                                 data.xpos[3 * chassisBody + 2]};
      const Eigen::Vector3d displacement = base - initialBase;
      const Eigen::Vector2d terrainPosition{displacement.dot(tangentX), displacement.dot(tangentY)};
      const double yaw = wrapAngle(yawInTerrain(data, chassisBody, tangentX, tangentY) - initialYaw);
      const double trajectoryError = (terrainPosition - odometry.head<2>()).norm();
      const double yawError = wrapAngle(yaw - odometry.z());
      const ControllerReference reference = readControllerReference(simulation);
      const Eigen::Vector3d targetDisplacement = reference.position - initialBase;
      const Eigen::Vector2d targetPosition{targetDisplacement.dot(tangentX), targetDisplacement.dot(tangentY)};
      const double targetYaw = wrapAngle(reference.yaw - initialYaw);
      const double trackingPositionError = (terrainPosition - targetPosition).norm();
      const double trackingYawError = wrapAngle(yaw - targetYaw);
      const double cycleWallMilliseconds =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cycleStart).count();
      finalPosition = terrainPosition;
      finalYaw = yaw;
      if(cycle >= options.warmupCycles)
      {
        updateStatistics(statistics, samplesWithModes, contactFallback, trajectoryError, yawError,
                         trackingPositionError, trackingYawError, cycleWallMilliseconds);
      }
      writeCsvRow(csv, cycle, data, base, terrainPosition, yaw, odometry, trajectoryError, yawError, targetPosition,
                  targetYaw, trackingPositionError, trackingYawError, reference, cycleWallMilliseconds, contactFallback,
                  samplesWithModes);
      ++completedCycles;
    }
    const std::clock_t cpuEnd = std::clock();
    const auto wallEnd = std::chrono::steady_clock::now();
    const double wallSeconds = std::chrono::duration<double>(wallEnd - wallStart).count();
    const double cpuSeconds = static_cast<double>(cpuEnd - cpuStart) / static_cast<double>(CLOCKS_PER_SEC);
    rusage usage{};
    if(getrusage(RUSAGE_SELF, &usage) != 0) { throw std::runtime_error("getrusage failed"); }
    writeReport(options, statistics, completedCycles, staticParity, controllerFailed, allFinite, data.time, wallSeconds,
                cpuSeconds, usage.ru_maxrss, odometry, finalPosition, finalYaw, finalContactFallback, finalSamples);
    simulation.stopSimulation();
    if(controllerFailed || !allFinite || completedCycles != options.cycles) { return EXIT_FAILURE; }
    return EXIT_SUCCESS;
  }
  catch(const std::exception & exception)
  {
    std::cerr << "rolling_contact_mujoco_runner: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
}
