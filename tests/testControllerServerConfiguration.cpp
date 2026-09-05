/*
 * Copyright 2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include <mc_control/ControllerServerConfiguration.h>
#include <mc_rtc/Configuration.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(ScalarSocketConfigurationDisablesProtocol)
{
  const auto config = mc_rtc::Configuration::fromYAMLData(R"(
Timestep: 0.005
IPC: false
TCP:
  Host: "127.0.0.1"
  Ports: [45242, 45343]
WS: false
)");

  const auto server = mc_control::ControllerServerConfiguration::fromConfiguration(config);

  BOOST_CHECK(!server.ipc_socket);
  BOOST_CHECK(!server.websocket_config);
  BOOST_REQUIRE(server.tcp_config);
  BOOST_CHECK_EQUAL(server.tcp_config->host, "127.0.0.1");
  BOOST_CHECK_EQUAL(server.tcp_config->pub_port, 45242);
  BOOST_CHECK_EQUAL(server.tcp_config->pull_port, 45343);

  const auto pub_uris = server.pub_uris();
  const auto pull_uris = server.pull_uris();
  BOOST_REQUIRE_EQUAL(pub_uris.size(), 1);
  BOOST_REQUIRE_EQUAL(pull_uris.size(), 1);
  BOOST_CHECK_EQUAL(pub_uris.front(), "tcp://127.0.0.1:45242");
  BOOST_CHECK_EQUAL(pull_uris.front(), "tcp://127.0.0.1:45343");
}

