#include "screen_ocr/result_mapper.hpp"

#include <limits>

namespace screen_ocr
{
namespace
{

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
  msg.magnetic_field.x = 0.0;
  msg.magnetic_field.y = 0.0;
  msg.magnetic_field.z = 0.0;
  msg.magnetic_field_covariance = {
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
  };
  msg.signal_strength = 0.0f;

  msg.signal_strength_percent = to_nan_if_missing(recognition.signal_strength_percent);
  msg.depth_meters = to_nan_if_missing(recognition.depth_meters);
  msg.current_milliamps = to_nan_if_missing(recognition.current_milliamps);
  msg.pipeline_heading_degrees = to_nan_if_missing(recognition.pipeline_heading_degrees);
  msg.left_arrow = recognition.left_arrow;
  msg.right_arrow = recognition.right_arrow;
  return msg;
}

}  // namespace screen_ocr
