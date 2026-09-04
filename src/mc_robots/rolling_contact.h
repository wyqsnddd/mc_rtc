/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include <mc_rbdyn/RobotModule.h>

#include "api.h"

namespace mc_robots
{

/** Robot module for the tracked rolling-contact validation robots and Ranger Mini V3 model. */
struct MC_ROBOTS_DLLAPI RollingContactRobotModule : public mc_rbdyn::RobotModule
{
  RollingContactRobotModule(const std::string & descriptionPath, const std::string & variant);
};

} // namespace mc_robots
