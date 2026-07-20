#include "screen_ocr/ocr_client.hpp"

#include <curl/curl.h>

#include <nlohmann/json.hpp>

#include <memory>
#include <string>

namespace screen_ocr
{
namespace
{

size_t write_callback(char * contents, size_t size, size_t nmemb, void * userp)
{
  const size_t total = size * nmemb;
  auto * buffer = static_cast<std::string *>(userp);
  buffer->append(contents, total);
  return total;
}

std::string trim_trailing_slash(std::string url)
{
  while (!url.empty() && url.back() == '/') {
    url.pop_back();
  }
  return url;
}

std::optional<double> read_optional_number(const nlohmann::json & payload, const char * key)
{
  if (!payload.contains(key) || payload[key].is_null()) {
    return std::nullopt;
  }
  if (payload[key].is_number()) {
    return payload[key].get<double>();
  }
  return std::nullopt;
}

std::optional<RecognitionResult> parse_flat_response(const nlohmann::json & payload)
{
  if (!payload.is_object()) {
    return std::nullopt;
  }

  if (!payload.contains("signal_strength_percent") &&
    !payload.contains("current_milliamps") &&
    !payload.contains("depth_meters") &&
    !payload.contains("pipeline_heading_degrees"))
  {
    return std::nullopt;
  }

  RecognitionResult result;
  result.signal_strength_percent = read_optional_number(payload, "signal_strength_percent");
  result.depth_meters = read_optional_number(payload, "depth_meters");
  result.current_milliamps = read_optional_number(payload, "current_milliamps");
  result.pipeline_heading_degrees = read_optional_number(payload, "pipeline_heading_degrees");
  result.left_arrow = payload.value("left_arrow", false);
  result.right_arrow = payload.value("right_arrow", false);
  return result;
}

std::optional<RecognitionResult> parse_legacy_response(const nlohmann::json & payload)
{
  if (!payload.value("success", false) || !payload.contains("data") || !payload["data"].is_object()) {
    return std::nullopt;
  }

  const auto & data = payload["data"];
  RecognitionResult result;
  result.left_arrow = false;
  result.right_arrow = false;

  if (data.contains("compass_angle_deg") && data["compass_angle_deg"].is_number()) {
    result.pipeline_heading_degrees = data["compass_angle_deg"].get<double>();
  }

  if (data.contains("signal_strength") && data["signal_strength"].is_string()) {
    const auto text = data["signal_strength"].get<std::string>();
    if (!text.empty() && text != "none") {
      try {
        result.signal_strength_percent = std::stod(text);
      } catch (const std::exception &) {
        // Keep optional empty.
      }
    }
  }

  if (data.contains("pipeline_current") && data["pipeline_current"].is_string()) {
    const auto text = data["pipeline_current"].get<std::string>();
    if (!text.empty() && text != "none") {
      try {
        result.current_milliamps = std::stod(text);
      } catch (const std::exception &) {
        // Keep optional empty.
      }
    }
  }

  if (data.contains("burial_depth") && data["burial_depth"].is_string()) {
    const auto text = data["burial_depth"].get<std::string>();
    if (!text.empty() && text != "none") {
      try {
        result.depth_meters = std::stod(text);
      } catch (const std::exception &) {
        // Keep optional empty.
      }
    }
  }

  if (data.contains("arrow_direction") && data["arrow_direction"].is_string()) {
    const auto direction = data["arrow_direction"].get<std::string>();
    result.left_arrow = direction == "left" || direction == "both";
    result.right_arrow = direction == "right" || direction == "both";
  }

  return result;
}

std::optional<RecognitionResult> parse_recognition_json(const std::string & body)
{
  nlohmann::json payload;
  try {
    payload = nlohmann::json::parse(body);
  } catch (const nlohmann::json::exception &) {
    return std::nullopt;
  }

  if (auto flat = parse_flat_response(payload)) {
    return flat;
  }
  return parse_legacy_response(payload);
}

class CurlGlobalInit
{
public:
  CurlGlobalInit() { curl_global_init(CURL_GLOBAL_DEFAULT); }
  ~CurlGlobalInit() { curl_global_cleanup(); }
};

}  // namespace

std::optional<RecognitionResult> recognize_image(
  const std::vector<uint8_t> & image_bytes,
  const std::string & api_base_url,
  const std::string & frame_id,
  bool debug,
  double timeout_sec)
{
  static CurlGlobalInit curl_init;

  const std::string url = trim_trailing_slash(api_base_url) + "/v1/recognize";
  const std::string debug_value = debug ? "true" : "false";

  CURL * curl = curl_easy_init();
  if (curl == nullptr) {
    return std::nullopt;
  }

  std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl_guard(curl, curl_easy_cleanup);
  curl_mime * mime = curl_mime_init(curl);
  if (mime == nullptr) {
    return std::nullopt;
  }

  std::unique_ptr<curl_mime, decltype(&curl_mime_free)> mime_guard(mime, curl_mime_free);

  curl_mimepart * frame_part = curl_mime_addpart(mime);
  curl_mime_name(frame_part, "frame_id");
  curl_mime_data(frame_part, frame_id.c_str(), CURL_ZERO_TERMINATED);

  curl_mimepart * debug_part = curl_mime_addpart(mime);
  curl_mime_name(debug_part, "debug");
  curl_mime_data(debug_part, debug_value.c_str(), CURL_ZERO_TERMINATED);

  curl_mimepart * image_part = curl_mime_addpart(mime);
  curl_mime_name(image_part, "image");
  curl_mime_filename(image_part, "frame.jpg");
  curl_mime_type(image_part, "image/jpeg");
  curl_mime_data(
    image_part,
    reinterpret_cast<const char *>(image_bytes.data()),
    image_bytes.size());

  std::string response_body;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, static_cast<long>(timeout_sec));
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, static_cast<long>(timeout_sec));
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

  const CURLcode code = curl_easy_perform(curl);
  if (code != CURLE_OK) {
    return std::nullopt;
  }

  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  if (http_code < 200 || http_code >= 300) {
    return std::nullopt;
  }

  return parse_recognition_json(response_body);
}

}  // namespace screen_ocr
