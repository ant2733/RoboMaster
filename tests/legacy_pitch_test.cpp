#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "serial/serial.h"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace
{
constexpr uint8_t kHeader1 = 0x09;
constexpr uint8_t kHeader2 = 0x14;
constexpr uint8_t kTail = 0x18;

// 打包步兵云台当前已烧录代码(Visual.c)的 7 字节大端帧:
// [0x09, 0x14, yaw_H, yaw_L, pitch_H, pitch_L, 0x18], 单位: 0.01 度
// 当前云台仅测试上下(Pitch)轴, yaw 固定锁死为 0
std::array<uint8_t, 7> pack_legacy_pitch_frame(float pitch_deg)
{
  const float clamped = std::clamp(pitch_deg, -327.67f, 327.67f);
  const int16_t yaw_raw = 0;
  const int16_t pitch_raw = static_cast<int16_t>(std::lround(clamped * 100.0f));

  const uint16_t yaw_u16 = static_cast<uint16_t>(yaw_raw);
  const uint16_t pitch_u16 = static_cast<uint16_t>(pitch_raw);

  return {
    kHeader1,
    kHeader2,
    static_cast<uint8_t>((yaw_u16 >> 8) & 0xFF),
    static_cast<uint8_t>(yaw_u16 & 0xFF),
    static_cast<uint8_t>((pitch_u16 >> 8) & 0xFF),
    static_cast<uint8_t>(pitch_u16 & 0xFF),
    kTail};
}

std::string format_hex(const std::array<uint8_t, 7> & frame)
{
  return fmt::format(
    "{:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X}", frame[0], frame[1], frame[2], frame[3],
    frame[4], frame[5], frame[6]);
}

bool send_pitch_deg(
  serial::Serial * ser, float pitch_deg, float max_pitch_deg, bool dry_run, bool log_frame)
{
  const float safe_pitch = std::clamp(pitch_deg, -max_pitch_deg, max_pitch_deg);
  const auto frame = pack_legacy_pitch_frame(safe_pitch);

  if (dry_run || ser == nullptr) {
    if (log_frame) {
      tools::logger()->info(
        "[DRY-RUN] yaw=0.00 deg, pitch={:+.2f} deg | HEX: {}", safe_pitch, format_hex(frame));
    }
    return true;
  }

  try {
    const size_t written = ser->write(frame.data(), frame.size());
    if (written != frame.size()) {
      tools::logger()->warn("Serial wrote {}/{} bytes", written, frame.size());
      return false;
    }
    if (log_frame) {
      tools::logger()->info(
        "[TX] yaw=0.00 deg, pitch={:+.2f} deg | HEX: {}", safe_pitch, format_hex(frame));
    }
    return true;
  } catch (const std::exception & e) {
    tools::logger()->error("Serial write failed: {}", e.what());
    return false;
  }
}

void run_manual_mode(serial::Serial * ser, float max_pitch_deg, bool dry_run)
{
  tools::logger()->info("===== 手动 Pitch 单轴测试模式 =====");
  tools::logger()->info("注意: 云台端需要按住鼠标右键(Remote_Mouse_KeyR==1)才会响应视觉帧!");
  tools::logger()->info(
    "输入数字(单位: 度, 限幅 +/-{:.1f} deg)按回车发送一帧; 快捷指令: w(+2度), s(-2度), 0(0度), q(退出)",
    max_pitch_deg);

  std::string line;
  while (true) {
    std::cout << "pitch_deg> " << std::flush;
    if (!std::getline(std::cin, line)) break;

    if (line == "q" || line == "Q" || line == "exit") break;
    if (line.empty()) continue;

    float pitch_deg = 0.0f;
    if (line == "w" || line == "W") {
      pitch_deg = std::min(2.0f, max_pitch_deg);
    } else if (line == "s" || line == "S") {
      pitch_deg = -std::min(2.0f, max_pitch_deg);
    } else {
      try {
        pitch_deg = std::stof(line);
      } catch (...) {
        tools::logger()->warn("无效输入 '{}', 请输入数字(如 2 / -2 / 0)或 w / s / q", line);
        continue;
      }
    }

    send_pitch_deg(ser, pitch_deg, max_pitch_deg, dry_run, true);
  }
}

void run_camera_mode(
  serial::Serial * ser, const std::string & config_path, float max_pitch_deg, float gain,
  float deadband_deg, int interval_ms, bool invert_pitch, bool dry_run)
{
  tools::logger()->info("===== 相机 + Pitch 单轴闭环测试模式 =====");
  tools::logger()->info(
    "config={}, max_pitch=+/-{:.1f} deg, gain={:.2f}, deadband={:.2f} deg, interval={} ms, invert={}",
    config_path, max_pitch_deg, gain, deadband_deg, interval_ms, invert_pitch);
  tools::logger()->info("预览窗口快捷键: 空格=暂停/恢复发送, q或ESC=退出");

  tools::Exiter exiter;
  io::Camera camera(config_path);
  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  // 没有云台回传四元数时，保持单位姿态，直接使用云台系 xyz_in_gimbal
  solver.set_R_gimbal2world(Eigen::Quaterniond::Identity());

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  auto last_tx_time = std::chrono::steady_clock::now() - std::chrono::milliseconds(interval_ms);
  bool tx_enabled = true;
  int frame_count = 0;

  cv::namedWindow("legacy_pitch_test", cv::WINDOW_NORMAL);

  while (!exiter.exit()) {
    camera.read(img, timestamp);
    if (img.empty()) continue;

    auto armors = yolo.detect(img, frame_count++);

    // 选一个离画面中心最近的装甲板
    auto_aim::Armor * best_armor = nullptr;
    double best_center_dist = std::numeric_limits<double>::max();
    const cv::Point2f img_center(img.cols * 0.5f, img.rows * 0.5f);

    for (auto & armor : armors) {
      solver.solve(armor);
      const double d = cv::norm(armor.center - img_center);
      if (d < best_center_dist) {
        best_center_dist = d;
        best_armor = &armor;
      }
    }

    cv::Mat vis = img.clone();
    cv::line(vis, {img.cols / 2 - 20, img.rows / 2}, {img.cols / 2 + 20, img.rows / 2}, {0, 255, 0}, 1);
    cv::line(vis, {img.cols / 2, img.rows / 2 - 20}, {img.cols / 2, img.rows / 2 + 20}, {0, 255, 0}, 1);

    if (best_armor != nullptr) {
      for (size_t i = 0; i < best_armor->points.size(); ++i) {
        cv::line(
          vis, best_armor->points[i], best_armor->points[(i + 1) % best_armor->points.size()],
          {0, 255, 255}, 2);
      }

      // xyz_in_gimbal: x前, y左, z上 -> ypd[1] 为相对光轴仰角(目标偏上为正)
      const Eigen::Vector3d ypd = tools::xyz2ypd(best_armor->xyz_in_gimbal);
      const float raw_pitch_deg = static_cast<float>(ypd[1] * 180.0 / CV_PI);
      const float dist_m = static_cast<float>(ypd[2]);

      float cmd_pitch_deg = 0.0f;
      if (std::abs(raw_pitch_deg) >= deadband_deg) {
        cmd_pitch_deg = raw_pitch_deg * gain;
        if (invert_pitch) cmd_pitch_deg = -cmd_pitch_deg;
        cmd_pitch_deg = std::clamp(cmd_pitch_deg, -max_pitch_deg, max_pitch_deg);
      }

      const auto now = std::chrono::steady_clock::now();
      const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_tx_time).count();
      if (tx_enabled && std::abs(raw_pitch_deg) >= deadband_deg && elapsed_ms >= interval_ms) {
        send_pitch_deg(ser, cmd_pitch_deg, max_pitch_deg, dry_run, true);
        last_tx_time = now;
      }

      cv::putText(
        vis,
        fmt::format(
          "dist={:.2f}m raw_pitch={:+.2f}deg cmd_pitch={:+.2f}deg TX={}", dist_m, raw_pitch_deg,
          cmd_pitch_deg, tx_enabled ? "ON" : "PAUSED"),
        {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2);
    } else {
      cv::putText(
        vis, fmt::format("NO TARGET | TX={}", tx_enabled ? "ON" : "PAUSED"), {20, 40},
        cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 0, 255}, 2);
    }

    cv::imshow("legacy_pitch_test", vis);
    const int key = cv::waitKey(1);
    if (key == 'q' || key == 'Q' || key == 27) break;
    if (key == ' ') {
      tx_enabled = !tx_enabled;
      tools::logger()->info("串口发送状态切换为: {}", tx_enabled ? "ON" : "PAUSED");
    }
  }
}

}  // namespace

const std::string keys =
  "{help h usage ? |                    | 输出命令行参数说明}"
  "{mode m         | manual             | 运行模式: manual(纯串口手动测试) 或 camera(相机Pitch跟瞄)}"
  "{port p         | /dev/ttyUSB0       | 串口设备路径}"
  "{baud b         | 115200             | 波特率(云台Visual.c默认115200)}"
  "{config c       | configs/demo.yaml  | camera模式使用的yaml配置文件}"
  "{max-pitch      | 5.0                | 单帧Pitch最大绝对值限幅(度)}"
  "{gain           | 0.35               | camera模式比例阻尼系数(防超调震荡)}"
  "{deadband       | 0.5                | camera模式死区(度)}"
  "{interval       | 60                 | camera模式最小发包间隔(ms)}"
  "{invert         | false              | 是否对Pitch符号取反}"
  "{dry-run        | false              | 仅打印发包内容, 不打开真实串口}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const auto mode = cli.get<std::string>("mode");
  const auto port = cli.get<std::string>("port");
  const auto baud = static_cast<uint32_t>(cli.get<int>("baud"));
  const auto config_path = cli.get<std::string>("config");
  const auto max_pitch_deg = std::abs(cli.get<float>("max-pitch"));
  const auto gain = cli.get<float>("gain");
  const auto deadband_deg = std::abs(cli.get<float>("deadband"));
  const auto interval_ms = std::max(10, cli.get<int>("interval"));
  const auto invert_pitch = cli.get<bool>("invert");
  const auto dry_run = cli.get<bool>("dry-run");

  std::unique_ptr<serial::Serial> ser;
  if (!dry_run) {
    try {
      ser = std::make_unique<serial::Serial>(port, baud, serial::Timeout::simpleTimeout(100));
      tools::logger()->info("Opened serial port {} @ {} bps", port, baud);
    } catch (const std::exception & e) {
      tools::logger()->error("Failed to open serial {}: {}", port, e.what());
      return 1;
    }
  } else {
    tools::logger()->info("Running in DRY-RUN mode (serial port not opened).");
  }

  if (mode == "manual") {
    run_manual_mode(ser.get(), max_pitch_deg, dry_run);
  } else if (mode == "camera") {
    run_camera_mode(
      ser.get(), config_path, max_pitch_deg, gain, deadband_deg, interval_ms, invert_pitch,
      dry_run);
  } else {
    tools::logger()->error("Unknown mode '{}'. Use --mode=manual or --mode=camera", mode);
    return 1;
  }

  return 0;
}
