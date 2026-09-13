/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include <mc_rbdyn/RobotModule.h>

#include "api.h"

namespace mc_robots
{

/** Robot module for the Ranger Mini V3 model and the tracked rolling-contact validation robots. */
struct MC_ROBOTS_DLLAPI RangerMiniV3RobotModule : public mc_rbdyn::RobotModule
{
  RangerMiniV3RobotModule(const std::string & descriptionPath, const std::string & variant);
};

} // namespace mc_robots
