#include <mc_control/Ticker.h>

#include <mc_rtc/Configuration.h>

#include <boost/program_options.hpp>
#include <spdlog/spdlog.h>

#include <exception>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace po = boost::program_options;
namespace fs = std::filesystem;

namespace
{

/** A temporary file that is removed when the ticker exits. */
struct ScopedTemporaryFile
{
  ScopedTemporaryFile() = default;
  ScopedTemporaryFile(const ScopedTemporaryFile &) = delete;
  ScopedTemporaryFile & operator=(const ScopedTemporaryFile &) = delete;
  ~ScopedTemporaryFile()
  {
    if(!path.empty())
    {
      std::error_code ignored;
      fs::remove(path, ignored);
    }
  }
  fs::path path;
};

/** Translate roslaunch-style "key:=value" tokens into an mc_rtc configuration.
 *
 * Only the keys that map cleanly onto a global configuration entry are
 * accepted. Anything else is a hard error: a silently ignored override looks
 * exactly like a successful run, which is how "-s robot:=..." went unnoticed.
 */
mc_rtc::Configuration parseOverrides(const std::vector<std::string> & tokens)
{
  mc_rtc::Configuration overrides;
  for(const auto & token : tokens)
  {
    const auto separator = token.find(":=");
    if(separator == std::string::npos)
    {
      mc_rtc::log::error_and_throw("Unexpected argument \"{}\": expected a key:=value override, one of robot:=NAME "
                                   "or controller:=NAME",
                                   token);
    }
    const std::string key = token.substr(0, separator);
    const std::string value = token.substr(separator + 2);
    if(value.empty()) { mc_rtc::log::error_and_throw("Override \"{}\" has an empty value", token); }
    if(key == "robot") { overrides.add("MainRobot", value); }
    else if(key == "controller")
    {
      overrides.add("Enabled", std::vector<std::string>{value});
      overrides.add("Default", value);
    }
    else
    {
      mc_rtc::log::error_and_throw("Unknown override key \"{}\" in \"{}\": only robot:= and controller:= are "
                                   "supported",
                                   key, token);
    }
  }
  return overrides;
}

} // namespace

int main(int argc, char * argv[])
{
  mc_control::Ticker::Configuration config;
  // Kept alive for the whole run: the ticker reads the file lazily.
  ScopedTemporaryFile overrideFile;
  // A bad command line is a user error, not a crash: report it and exit rather
  // than letting the exception reach std::terminate and dump core.
  try
  {
    bool replay_outputs = false;
    bool only_gui_inputs = false;
    bool continue_after_replay = false;
    std::vector<std::string> overrides;
    po::options_description desc("mc_rtc_ticker options");
    // clang-format off
    desc.add_options()
      ("help", "Show this help message")
      ("mc-config,f", po::value<std::string>(&config.mc_rtc_configuration), "Configuration given to mc_rtc")
      ("step-by-step,S", po::bool_switch(&config.step_by_step), "Start the ticker in step-by-step mode")
      ("run-for", po::value<double>(&config.run_for), "Run for the specified time (seconds)")
      ("no-sync,s", po::bool_switch(&config.no_sync), "Synchronize ticker time with real time")
      ("sync-ratio,r", po::value<double>(&config.sync_ratio), "Sim/real ratio for synchronization purpose")
      ("replay-log,l", po::value<std::string>(&config.replay_configuration.log), "Log to replay")
      ("datastore-mapping,m", po::value<std::string>(&config.replay_configuration.with_datastore_config), "Mapping of log keys to datastore")
      ("replay-gui-inputs-only,g", po::bool_switch(&only_gui_inputs), "Only replay the GUI inputs")
      ("continue-after-replay,c", po::bool_switch(&continue_after_replay), "Continue after log replay")
      ("exit-after-replay,e", po::bool_switch(&config.replay_configuration.exit_after_log), "Exit after log replay")
      ("replay-outputs", po::bool_switch(&replay_outputs), "Enable outputs replay (override controller)")
      ("override", po::value<std::vector<std::string>>(&overrides), "roslaunch-style override, robot:=NAME or controller:=NAME");
    // clang-format on
    po::positional_options_description positional;
    positional.add("override", -1);
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(positional).run(), vm);
    po::notify(vm);
    if(vm.count("help"))
    {
      std::cout << desc << "\n";
      return 0;
    }
    if(replay_outputs) { config.replay_configuration.with_outputs = true; }
    if(only_gui_inputs)
    {
      if(replay_outputs)
      {
        mc_rtc::log::error_and_throw("--replay-outputs is contradictory with --replay-gui-inputs-only");
      }
      config.replay_configuration.with_inputs = false;
      config.replay_configuration.with_outputs = false;
    }
    if(config.sync_ratio <= 0) { mc_rtc::log::error_and_throw("sync-ratio must be strictly positive"); }
    config.replay_configuration.stop_after_log = !continue_after_replay;
    if(!overrides.empty())
    {
      // The overrides have to reach mc_rtc as the -f configuration rather than
      // through MC_RTC_CONTROLLER_CONFIG: GlobalConfiguration merges the
      // environment files *before* the user configuration, so an override put
      // there would be silently beaten by ~/.config/mc_rtc/mc_rtc.yaml. The -f
      // configuration is merged last and therefore wins.
      mc_rtc::Configuration merged;
      if(!config.mc_rtc_configuration.empty() && fs::exists(config.mc_rtc_configuration))
      {
        merged.load(config.mc_rtc_configuration);
      }
      merged.load(parseOverrides(overrides));
      std::random_device entropy;
      overrideFile.path =
          fs::temp_directory_path() / ("mc_rtc_ticker-overrides-" + std::to_string(entropy()) + ".yaml");
      merged.save(overrideFile.path.string());
      config.mc_rtc_configuration = overrideFile.path.string();
    }
  }
  catch(const std::exception & exc)
  {
    // error_and_throw has already logged the details; keep this to one line.
    mc_rtc::log::error("mc_rtc_ticker: {}", exc.what());
    return 1;
  }
  mc_control::Ticker ticker(config);
  ticker.run();
  // ROS 2's spdlog backend may register a periodic worker whose callbacks live
  // in a dynamically loaded plugin. Stop it before ticker destruction unloads
  // those plugins, rather than leaving it to static destruction at process exit.
  spdlog::shutdown();
  return 0;
}
