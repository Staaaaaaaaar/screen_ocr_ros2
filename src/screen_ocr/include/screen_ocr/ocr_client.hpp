#pragma once

#include <optional>
#include <string>
#include <vector>

namespace screen_ocr
{

struct RecognitionResult
{
  std::optional<double> signal_strength_percent;
  std::optional<double> depth_meters;
  std::optional<double> current_milliamps;
  std::optional<double> pipeline_heading_degrees;
  bool left_arrow{false};
  bool right_arrow{false};
};

std::optional<RecognitionResult> recognize_image(
  const std::vector<uint8_t> & image_bytes,
  const std::string & api_base_url,
  const std::string & frame_id,
  bool debug,
  double timeout_sec);

}  // namespace screen_ocr
