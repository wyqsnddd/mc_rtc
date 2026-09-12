/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include "rolling_contact.h"

#include <mc_rbdyn/BodySensor.h>
#include <mc_rbdyn/RobotModuleMacros.h>

#include <mc_rtc/logging.h>

#include <RBDyn/parsers/urdf.h>

namespace mc_robots
{

RollingContactRobotModule::RollingContactRobotModule(const std::string & descriptionPath, const std::string & variant)
: RobotModule(descriptionPath, variant)
{
  if(variant != "rolling_diff" && variant != "rolling_4s" && variant != "ranger_mini_v3")
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>("Unknown rolling-contact robot variant: {}", variant);
  }

  init(rbd::parsers::from_urdf_file(urdf_path, rbd::parsers::ParserParameters{}.fixed(false)));
  _canonicalParameters = {"RollingContact", descriptionPath, variant};
  const bool rangerMiniV3 = variant == "ranger_mini_v3";
  // Ranger's floating frame is at the geometric centre of the chassis shell,
  // coincident with the link/inertial origin and the MuJoCo free-joint body
  // frame. The wheel axle plane is 35 mm below it; the URDF and MuJoCo
  // descriptions apply the inverse offset to their wheel bodies so the
  // contact geometry remains unchanged when the base frame is rebased.
  _default_attitude = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, rangerMiniV3 ? 0.16 : 0.2}};
  _bodySensors.emplace_back("FloatingBase", "chassis", sva::PTransformd::Identity());
  if(rangerMiniV3)
  {
    // Coincident with the MuJoCo <site name="imu" pos="0 0 0"/> in
    // ranger_mini_v3.xml. The two must stay equal: X_b_s is what the tilt
    // observer will use to rotate the gyro into the chassis frame and to add the
    // accelerometer's lever arm, and mc_mujoco reports the site's own frame.
    _bodySensors.emplace_back("ChassisIMU", "chassis", sva::PTransformd::Identity());
  }

  if(variant == "rolling_diff")
  {
    _ref_joint_order = {"left_drive", "right_drive"};
    _frames.emplace_back("left_carrier", "chassis", sva::PTransformd(Eigen::Vector3d(0.0, 0.3, 0.0)));
    _frames.emplace_back("right_carrier", "chassis", sva::PTransformd(Eigen::Vector3d(0.0, -0.3, 0.0)));
  }
  else
  {
    _ref_joint_order = {"front_left_steer", "front_left_drive", "front_right_steer", "front_right_drive",
                        "rear_left_steer",  "rear_left_drive",  "rear_right_steer",  "rear_right_drive"};
    _frames.emplace_back("front_left_carrier", "front_left_knuckle", sva::PTransformd::Identity());
    _frames.emplace_back("front_right_carrier", "front_right_knuckle", sva::PTransformd::Identity());
    _frames.emplace_back("rear_left_carrier", "rear_left_knuckle", sva::PTransformd::Identity());
    _frames.emplace_back("rear_right_carrier", "rear_right_knuckle", sva::PTransformd::Identity());
  }
  for(const auto & joint : _ref_joint_order) { _stance[joint] = {0.0}; }
}

} // namespace mc_robots

#ifndef MC_RTC_BUILD_STATIC

extern "C"
{
  ROBOT_MODULE_API void MC_RTC_ROBOT_MODULE(std::vector<std::string> & names)
  {
    ROBOT_MODULE_CHECK_VERSION("RollingContact")
    names = {"RollingContact"};
  }

  ROBOT_MODULE_API void destroy(mc_rbdyn::RobotModule * ptr)
  {
    delete ptr;
  }

  ROBOT_MODULE_API mc_rbdyn::RobotModule * create(const std::string &,
                                                  const std::string & descriptionPath,
                                                  const std::string & variant)
  {
    return new mc_robots::RollingContactRobotModule(descriptionPath, variant);
  }
}

#else

#  include <mc_rbdyn/RobotLoader.h>

namespace
{

static auto registered = []()
{
  using fn_t = std::function<mc_robots::RollingContactRobotModule *(const std::string &, const std::string &)>;
  mc_rbdyn::RobotLoader::register_object(
      "RollingContact", fn_t([](const std::string & path, const std::string & variant)
                             { return new mc_robots::RollingContactRobotModule(path, variant); }));
  return true;
}();

} // namespace

#endif
