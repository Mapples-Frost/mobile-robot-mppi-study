// Minimal Livox SDK2 -> localhost UDP bridge for the Raspberry Pi 5 runtime.
//
// The bridge performs no filtering and never talks to the vehicle CAN bus. It
// normalizes Cartesian point types into float32 metres while preserving the
// reflectivity/tag bytes required by the Python perception process.

#include "livox_lidar_api.h"
#include "livox_lidar_def.h"

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr char kMagic[4] = {'L', 'V', 'X', '1'};
constexpr std::uint16_t kVersion = 1;
constexpr std::uint16_t kDefaultPort = 57701;

#pragma pack(push, 1)
struct WireHeader {
  char magic[4];
  std::uint16_t version;
  std::uint16_t point_count;
  std::uint32_t sequence;
  std::uint64_t callback_timestamp_ns;
};

struct WirePoint {
  float x_m;
  float y_m;
  float z_m;
  std::uint8_t reflectivity;
  std::uint8_t tag;
};
#pragma pack(pop)

static_assert(sizeof(WireHeader) == 20, "unexpected wire header size");
static_assert(sizeof(WirePoint) == 14, "unexpected wire point size");

std::atomic<bool> g_running{true};
int g_socket = -1;
sockaddr_in g_destination{};
std::atomic<std::uint64_t> g_packets{0};
std::atomic<std::uint64_t> g_dropped{0};

std::uint64_t MonotonicNanoseconds() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void AppendHighPoints(const LivoxLidarEthernetPacket* packet,
                      std::vector<WirePoint>* output) {
  const auto* points = reinterpret_cast<const LivoxLidarCartesianHighRawPoint*>(
      packet->data);
  output->reserve(packet->dot_num);
  for (std::uint32_t index = 0; index < packet->dot_num; ++index) {
    const auto& point = points[index];
    output->push_back({point.x * 0.001F, point.y * 0.001F,
                       point.z * 0.001F, point.reflectivity, point.tag});
  }
}

void AppendLowPoints(const LivoxLidarEthernetPacket* packet,
                     std::vector<WirePoint>* output) {
  const auto* points = reinterpret_cast<const LivoxLidarCartesianLowRawPoint*>(
      packet->data);
  output->reserve(packet->dot_num);
  for (std::uint32_t index = 0; index < packet->dot_num; ++index) {
    const auto& point = points[index];
    output->push_back({point.x * 0.01F, point.y * 0.01F,
                       point.z * 0.01F, point.reflectivity, point.tag});
  }
}

void AppendDoubleEchoPoints(const LivoxLidarEthernetPacket* packet,
                            std::vector<WirePoint>* output) {
  const auto* points = reinterpret_cast<const LivoxLidarDoubleEchoRawPoint*>(
      packet->data);
  output->reserve(2U * packet->dot_num);
  for (std::uint32_t index = 0; index < packet->dot_num; ++index) {
    const auto& point = points[index];
    output->push_back({point.x1 * 0.001F, point.y1 * 0.001F,
                       point.z1 * 0.001F, point.reflectivity1, point.tag1});
    output->push_back({point.x2 * 0.001F, point.y2 * 0.001F,
                       point.z2 * 0.001F, point.reflectivity2, point.tag2});
  }
}

void PointCloudCallback(std::uint32_t, const std::uint8_t,
                        LivoxLidarEthernetPacket* packet, void*) {
  if (packet == nullptr || g_socket < 0 || !g_running.load()) {
    return;
  }
  std::vector<WirePoint> points;
  switch (packet->data_type) {
    case kLivoxLidarCartesianCoordinateHighData:
      AppendHighPoints(packet, &points);
      break;
    case kLivoxLidarCartesianCoordinateLowData:
      AppendLowPoints(packet, &points);
      break;
    case kLivoxLidarDoubleEchoData:
      AppendDoubleEchoPoints(packet, &points);
      break;
    default:
      return;
  }
  if (points.empty() || points.size() > UINT16_MAX) {
    return;
  }

  WireHeader header{};
  std::memcpy(header.magic, kMagic, sizeof(kMagic));
  header.version = kVersion;
  header.point_count = static_cast<std::uint16_t>(points.size());
  header.sequence = packet->udp_cnt;
  header.callback_timestamp_ns = MonotonicNanoseconds();
  std::vector<std::uint8_t> datagram(
      sizeof(header) + points.size() * sizeof(WirePoint));
  std::memcpy(datagram.data(), &header, sizeof(header));
  std::memcpy(datagram.data() + sizeof(header), points.data(),
              points.size() * sizeof(WirePoint));
  const ssize_t sent = sendto(
      g_socket, datagram.data(), datagram.size(), MSG_DONTWAIT,
      reinterpret_cast<const sockaddr*>(&g_destination),
      sizeof(g_destination));
  if (sent == static_cast<ssize_t>(datagram.size())) {
    ++g_packets;
  } else {
    ++g_dropped;
  }
}

void LidarInfoChangeCallback(std::uint32_t handle, const LivoxLidarInfo* info,
                             void*) {
  if (info != nullptr) {
    std::fprintf(stderr, "Livox ready: handle=%u serial=%s\n", handle,
                 info->sn);
  }
  SetLivoxLidarWorkMode(handle, kLivoxLidarNormal, nullptr, nullptr);
  SetLivoxLidarPclDataType(handle, kLivoxLidarCartesianCoordinateHighData,
                           nullptr, nullptr);
}

void StopHandler(int) { g_running.store(false); }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "usage: %s CONFIG_JSON [UDP_PORT]\n", argv[0]);
    return 2;
  }
  const int port = argc == 3 ? std::stoi(argv[2]) : kDefaultPort;
  if (port <= 0 || port > 65535) {
    std::fprintf(stderr, "invalid UDP port\n");
    return 2;
  }
  std::signal(SIGINT, StopHandler);
  std::signal(SIGTERM, StopHandler);

  g_socket = socket(AF_INET, SOCK_DGRAM, 0);
  if (g_socket < 0) {
    std::perror("socket");
    return 1;
  }
  g_destination.sin_family = AF_INET;
  g_destination.sin_port = htons(static_cast<std::uint16_t>(port));
  inet_pton(AF_INET, "127.0.0.1", &g_destination.sin_addr);

  if (!LivoxLidarSdkInit(argv[1])) {
    std::fprintf(stderr, "Livox SDK initialization failed\n");
    close(g_socket);
    return 1;
  }
  SetLivoxLidarPointCloudCallBack(PointCloudCallback, nullptr);
  SetLivoxLidarInfoChangeCallback(LidarInfoChangeCallback, nullptr);
  std::fprintf(stderr, "Streaming Livox points to 127.0.0.1:%d\n", port);

  std::uint64_t last_packets = 0;
  while (g_running.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    const std::uint64_t packets = g_packets.load();
    std::fprintf(stderr, "packets=%llu rate=%llu/s dropped=%llu\n",
                 static_cast<unsigned long long>(packets),
                 static_cast<unsigned long long>(packets - last_packets),
                 static_cast<unsigned long long>(g_dropped.load()));
    last_packets = packets;
  }
  LivoxLidarSdkUninit();
  close(g_socket);
  std::fprintf(stderr, "Livox bridge stopped cleanly\n");
  return 0;
}
