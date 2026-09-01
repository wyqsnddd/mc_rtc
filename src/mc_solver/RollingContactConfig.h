/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include <mc_rbdyn/RollingContact.h>

#include <mc_rtc/Configuration.h>

#include <stdexcept>
#include <vector>

namespace mc_solver::details
{

inline mc_rbdyn::RollingContactDescription loadRollingWheel(const mc_rtc::Configuration & config)
{
  mc_rbdyn::RollingContactDescription wheel;
  wheel.name = config("name", std::string{});
  wheel.carrierFrame = config("carrierFrame", std::string{});
  if(config.has("forceBody")) { wheel.wheelBody = config("forceBody", std::string{}); }
  else { wheel.wheelBody = config("wheelBody", std::string{}); }
  wheel.driveJoint = config("driveJoint", std::string{});
  wheel.steeringJoint = config("steeringJoint", std::string{});
  wheel.radius = config("radius", 0.0);
  wheel.width = config("width", 0.0);
  wheel.friction = config("friction", 0.7);
  wheel.spinSign = config("spinSign", 1.0);
  wheel.mode = mc_rbdyn::rollingContactModeFromString(config("mode", std::string{"rolling"}));
  wheel.activation = config("activation", 1.0);
  wheel.validate();
  return wheel;
}

inline std::vector<mc_rbdyn::RollingContactDescription> loadRollingWheels(const mc_rtc::Configuration & config)
{
  if(!config.has("wheels")) { throw std::invalid_argument("Rolling contact configuration requires a wheels array"); }
  std::vector<mc_rbdyn::RollingContactDescription> wheels;
  for(const auto & wheel : config("wheels")) { wheels.push_back(loadRollingWheel(wheel)); }
  if(wheels.empty()) { throw std::invalid_argument("Rolling contact wheels array cannot be empty"); }
  return wheels;
}

} // namespace mc_solver::details
