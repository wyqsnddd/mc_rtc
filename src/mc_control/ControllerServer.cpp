/*
 * Copyright 2015-2020 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include <mc_control/ControllerServer.h>

#ifndef MC_RTC_DISABLE_NETWORK
#  include <nanomsg/nn.h>
#  include <nanomsg/pipeline.h>
#  include <nanomsg/pubsub.h>
#  include <nanomsg/reqrep.h>
#endif

#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>

#if !defined(_WIN32) && !defined(MC_RTC_DISABLE_NETWORK)
#  include <sys/socket.h>
#  include <sys/un.h>
#  include <unistd.h>
#endif

namespace
{

/** Remove an IPC pathname left behind by a crashed server.
 *
 * nanomsg reports an address/permission error for both a live endpoint and a
 * stale Unix socket file. Probe the endpoint before unlinking it so that
 * starting a second controller cannot silently disconnect an already-running
 * controller.
 */
bool remove_stale_ipc_endpoint(const std::string & uri)
{
#if !defined(_WIN32) && !defined(MC_RTC_DISABLE_NETWORK)
  constexpr const char * prefix = "ipc://";
  if(uri.rfind(prefix, 0) != 0) { return false; }
  const std::string path = uri.substr(std::strlen(prefix));
  if(path.empty() || path.front() != '/') { return false; }

  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  if(path.size() >= sizeof(address.sun_path)) { return false; }
  std::memcpy(address.sun_path, path.c_str(), path.size() + 1);

  const int probe = ::socket(AF_UNIX, SOCK_STREAM, 0);
  int result = -1;
  int connect_errno = probe < 0 ? errno : 0;
  if(probe >= 0)
  {
    result = ::connect(probe, reinterpret_cast<const sockaddr *>(&address), sizeof(address));
    connect_errno = errno;
    ::close(probe);
  }
  else if(connect_errno != EPERM)
  {
    return false;
  }

  // ECONNREFUSED means the filesystem entry exists but no process is
  // listening. ENOENT handles a race with another cleanup operation.
  if(result == 0) { return false; }
  bool stale = connect_errno == ECONNREFUSED || connect_errno == ENOENT;
  if(!stale)
  {
    // Some restricted Linux environments reject a direct AF_UNIX probe with
    // EPERM even though nanomsg can bind the endpoint. Consult the kernel's
    // Unix-socket table in that case: a pathname present there belongs to a
    // live socket, while a regular filesystem entry left by a crashed process
    // is absent and can be removed safely. If /proc is unavailable, retain
    // the conservative behavior and leave the endpoint untouched.
    std::ifstream unixSockets{"/proc/net/unix"};
    if(!unixSockets) { return false; }
    std::string line;
    while(std::getline(unixSockets, line))
    {
      std::istringstream fields(line);
      std::string value;
      while(fields >> value)
      {
        if(value == path) { return false; }
      }
    }
    stale = true;
  }
  if(!stale) { return false; }
  if(::unlink(path.c_str()) == 0) { return true; }
  return errno == ENOENT;
#else
  static_cast<void>(uri);
  return false;
#endif
}

} // namespace

namespace mc_control
{

ControllerServer::ControllerServer(double dt, const ControllerServerConfiguration & config)
: ControllerServer(dt, config.timestep, config.pub_uris(), config.pull_uris())
{
}

ControllerServer::ControllerServer(double dt,
                                   double server_dt,
                                   const std::vector<std::string> & pub_bind_uri,
                                   const std::vector<std::string> & pull_bind_uri)
{
  iter_ = 0;
  update_rate(dt, server_dt);
#ifndef MC_RTC_DISABLE_NETWORK
  auto init_socket = [](int & socket, int proto, const std::vector<std::string> & uris, const std::string & name)
  {
    socket = nn_socket(AF_SP, proto);
    if(socket < 0) { mc_rtc::log::error_and_throw("Failed to initialize {}", name); }
    for(const auto & uri : uris)
    {
      int ret = nn_bind(socket, uri.c_str());
      // nanomsg may report EADDRINUSE, EACCES, or EPERM for a pathname that
      // is left behind by a crashed IPC endpoint. The cleanup helper only
      // removes an IPC filesystem entry after checking that no live Unix
      // socket owns it, so it is safe to try it for every bind failure.
      if(ret < 0 && remove_stale_ipc_endpoint(uri))
      {
        mc_rtc::log::warning("Removing stale IPC endpoint {}", uri);
        ret = nn_bind(socket, uri.c_str());
      }
      if(ret < 0) { mc_rtc::log::error_and_throw("Failed to bind {} to uri: {}", name, uri); }
    }
  };
  init_socket(pub_socket_, NN_PUB, pub_bind_uri, "PUB socket");
  init_socket(pull_socket_, NN_PULL, pull_bind_uri, "PULL socket");
#endif
}

ControllerServer::~ControllerServer()
{
#ifndef MC_RTC_DISABLE_NETWORK
  nn_close(pub_socket_);
  nn_close(pull_socket_);
#endif
}

void ControllerServer::handle_requests(mc_rtc::gui::StateBuilder & gui_builder, const char * dataIn)
{
  auto config = mc_rtc::Configuration::fromData(static_cast<const char *>(dataIn));
  auto category = config("category", std::vector<std::string>{});
  auto name = config("name", std::string{});
  auto data = config("data", mc_rtc::Configuration{});
  if(!gui_builder.handleRequest(category, name, data))
  {
    mc_rtc::log::error("Invokation of the following method failed\n{}\n", config.dump(true));
  }
  if(logger_) { logger_->addGUIEvent({std::move(category), std::move(name), data}); }
}

void ControllerServer::handle_requests(mc_rtc::gui::StateBuilder & gui_builder)
{
  for(auto & r : requests_)
  {
    if(!gui_builder.handleRequest(r.category, r.name, r.data))
    {
      mc_rtc::Configuration config;
      config.add("category", r.category);
      config.add("name", r.name);
      config.add("data", r.data);
      mc_rtc::log::error("Invokation of the following method failed\n{}\n", config.dump(true));
    }
    if(logger_) { logger_->addGUIEvent(std::move(r)); }
  }
  requests_.resize(0);
#ifndef MC_RTC_DISABLE_NETWORK
  /*FIXME Avoid freeing the message constantly */
  void * buf = nullptr;
  int recv = 0;
  do
  {
    recv = nn_recv(pull_socket_, &buf, NN_MSG, NN_DONTWAIT);
    if(recv < 0)
    {
      auto err = nn_errno();
      if(err != EAGAIN) { mc_rtc::log::error("ControllerServer failed to receive requested with errno: {}", err); }
    }
    else
    {
      handle_requests(gui_builder, static_cast<const char *>(buf));
      nn_freemsg(buf);
    }
  } while(recv > 0);
#endif
}

void ControllerServer::publish(mc_rtc::gui::StateBuilder & gui_builder)
{
  if(iter_++ % rate_ == 0)
  {
    buffer_size_ = gui_builder.update(buffer_);
#ifndef MC_RTC_DISABLE_NETWORK
    int err = nn_send(pub_socket_, buffer_.data(), buffer_size_, 0);
    if(err < 0) { mc_rtc::log::error("[ControllerServer] Failed to send {}", nn_strerror(nn_errno())); }
#endif
  }
  else
  {
    gui_builder.update();
    buffer_size_ = 0;
  }
}

std::pair<const char *, size_t> ControllerServer::data() const
{
  return {buffer_.data(), buffer_size_};
}

void ControllerServer::update_rate(double dt, double server_dt)
{
  if(server_dt < dt) { server_dt = dt; }
  rate_ = static_cast<unsigned int>(ceil(server_dt / dt));
}

} // namespace mc_control
