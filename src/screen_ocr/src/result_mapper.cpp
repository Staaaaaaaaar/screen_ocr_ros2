#include "screen_ocr/result_mapper.hpp"

#include <cmath>
#include <limits>

namespace screen_ocr
{
namespace
{

constexpr double kUnknownCovariance = -1.0;

geometry_msgs::msg::Vector3 heading_to_magnetic_field(std::optional<double> heading_deg)
{
  geometry_msgs::msg::Vector3 vector;
  if (!heading_deg.has_value()) {
    return vector;
  }

  const double radians = heading_deg.value() * M_PI / 180.0;
  vector.x = std::sin(radians);
  vector.y = std::cos(radians);
  vector.z = 0.0;
  return vector;
}

float to_nan_if_missing(const std::optional<double> & value)
{
  if (!value.has_value()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return static_cast<float>(value.value());
}

}  // namespace

screen_ocr_msgs::msg::SensorMsg recognition_to_sensor_msg(
  const RecognitionResult & recognition,
  const std_msgs::msg::Header & header)
{
  screen_ocr_msgs::msg::SensorMsg msg;
  msg.header = header;
  msg.magnetic_field = heading_to_magnetic_field(recognition.pipeline_heading_degrees);
  msg.magnetic_field_covariance = {
    kUnknownCovariance, 0.0, 0.0,
    0.0, kUnknownCovariance, 0.0,
    0.0, 0.0, kUnknownCovariance,
  };

  msg.signal_strength_percent = to_nan_if_missing(recognition.signal_strength_percent);
  msg.signal_strength = recognition.signal_strength_percent.has_value()
    ? static_cast<float>(recognition.signal_strength_percent.value() / 100.0)
    : std::numeric_limits<float>::quiet_NaN();
  msg.depth_meters = to_nan_if_missing(recognition.depth_meters);
  msg.current_milliamps = to_nan_if_missing(recognition.current_milliamps);
  msg.pipeline_heading_degrees = to_nan_if_missing(recognition.pipeline_heading_degrees);
  msg.left_arrow = recognition.left_arrow;
  msg.right_arrow = recognition.right_arrow;
  return msg;
}

}  // namespace screen_ocr
