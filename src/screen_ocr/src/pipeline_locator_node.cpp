#include "screen_ocr/ocr_client.hpp"
#include "screen_ocr/result_mapper.hpp"

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "screen_ocr_msgs/msg/sensor_msg.hpp"

namespace screen_ocr
{

class PipelineLocatorNode : public rclcpp::Node
{
public:
  PipelineLocatorNode()
  : Node("pipeline_locator_node")
  {
    declare_parameter("image_topic", "/image_raw");
    declare_parameter("image_type", "raw");
    declare_parameter("output_topic", "/pipeline_locator/sensor");
    declare_parameter("inference_rate_hz", 2.0);
    declare_parameter("output_frame_id", "pipeline_locator");
    declare_parameter("api_base_url", "http://127.0.0.1:8000");
    declare_parameter("api_timeout_sec", 5.0);
    declare_parameter("debug", false);
    declare_parameter("qos_reliability", "best_effort");
    declare_parameter("qos_history_depth", 1);

    image_topic_ = get_parameter("image_topic").as_string();
    image_type_ = get_parameter("image_type").as_string();
    output_topic_ = get_parameter("output_topic").as_string();
    inference_rate_hz_ = get_parameter("inference_rate_hz").as_double();
    output_frame_id_ = get_parameter("output_frame_id").as_string();
    api_base_url_ = get_parameter("api_base_url").as_string();
    api_timeout_sec_ = get_parameter("api_timeout_sec").as_double();
    debug_ = get_parameter("debug").as_bool();
    qos_reliability_ = get_parameter("qos_reliability").as_string();
    qos_history_depth_ = get_parameter("qos_history_depth").as_int();

    const auto qos = build_qos_profile();

    if (image_type_ == "compressed") {
      compressed_subscription_ = create_subscription<sensor_msgs::msg::CompressedImage>(
        image_topic_,
        qos,
        std::bind(&PipelineLocatorNode::compressed_image_callback, this, std::placeholders::_1));
    } else {
      image_subscription_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic_,
        qos,
        std::bind(&PipelineLocatorNode::image_callback, this, std::placeholders::_1));
    }

    publisher_ = create_publisher<screen_ocr_msgs::msg::SensorMsg>(output_topic_, 10);

    const double period_sec = 1.0 / std::max(inference_rate_hz_, 0.1);
    timer_ = create_wall_timer(
      std::chrono::duration<double>(period_sec),
      std::bind(&PipelineLocatorNode::on_timer, this));

    RCLCPP_INFO(
      get_logger(),
      "Pipeline locator node started: image=%s (%s), api=%s, output=%s, rate=%.2f Hz",
      image_topic_.c_str(),
      image_type_.c_str(),
      api_base_url_.c_str(),
      output_topic_.c_str(),
      inference_rate_hz_);
  }

private:
  rclcpp::QoS build_qos_profile()
  {
    rclcpp::QoS qos(std::max(qos_history_depth_, 1));
    if (qos_reliability_ == "reliable") {
      qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
    } else {
      qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
    }
    qos.durability(rclcpp::DurabilityPolicy::Volatile);
    return qos;
  }

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception & exc) {
      RCLCPP_WARN(get_logger(), "Failed to convert Image message: %s", exc.what());
      return;
    }

    std::lock_guard<std::mutex> lock(image_mutex_);
    latest_header_ = msg->header;
    latest_cv_image_ = cv_ptr->image.clone();
    latest_jpeg_bytes_.clear();
    has_latest_frame_ = true;
  }

  void compressed_image_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
  {
    if (!msg->data.empty()) {
      std::lock_guard<std::mutex> lock(image_mutex_);
      latest_header_ = msg->header;
      latest_jpeg_bytes_.assign(msg->data.begin(), msg->data.end());
      latest_cv_image_.release();
      has_latest_frame_ = true;
      return;
    }

    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception & exc) {
      RCLCPP_WARN(get_logger(), "Failed to convert CompressedImage message: %s", exc.what());
      return;
    }

    std::lock_guard<std::mutex> lock(image_mutex_);
    latest_header_ = msg->header;
    latest_cv_image_ = cv_ptr->image;
    latest_jpeg_bytes_.clear();
    has_latest_frame_ = true;
  }

  std::optional<std::vector<uint8_t>> encode_jpeg(const cv::Mat & image)
  {
    std::vector<uint8_t> encoded;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
    if (!cv::imencode(".jpg", image, encoded, params)) {
      return std::nullopt;
    }
    return encoded;
  }

  void on_timer()
  {
    std_msgs::msg::Header header;
    cv::Mat cv_image;
    std::vector<uint8_t> jpeg_bytes;
    bool has_frame = false;

    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      if (!has_latest_frame_) {
        RCLCPP_DEBUG(get_logger(), "Waiting for image on %s", image_topic_.c_str());
        return;
      }

      header = latest_header_;
      cv_image = latest_cv_image_;
      jpeg_bytes = latest_jpeg_bytes_;
      has_frame = true;
    }

    if (!has_frame) {
      return;
    }

    std::vector<uint8_t> request_bytes;
    if (!jpeg_bytes.empty()) {
      request_bytes = std::move(jpeg_bytes);
    } else if (!cv_image.empty()) {
      const auto encoded = encode_jpeg(cv_image);
      if (!encoded.has_value()) {
        RCLCPP_WARN(get_logger(), "Failed to encode image for frame %s", header.frame_id.c_str());
        return;
      }
      request_bytes = encoded.value();
    } else {
      RCLCPP_DEBUG(get_logger(), "Waiting for image on %s", image_topic_.c_str());
      return;
    }

    const auto recognition = recognize_image(
      request_bytes,
      api_base_url_,
      header.frame_id.empty() ? "ros" : header.frame_id,
      debug_,
      api_timeout_sec_);

    if (!recognition.has_value()) {
      RCLCPP_WARN(
        get_logger(),
        "Recognition failed for frame %s (api=%s)",
        header.frame_id.c_str(),
        api_base_url_.c_str());
      return;
    }

    std_msgs::msg::Header output_header = header;
    if (!output_frame_id_.empty()) {
      output_header.frame_id = output_frame_id_;
    }

    const auto sensor_msg = recognition_to_sensor_msg(recognition.value(), output_header);
    publisher_->publish(sensor_msg);

    RCLCPP_DEBUG(
      get_logger(),
      "Published heading=%.1f deg, current=%.0f mA, depth=%.0f m, signal=%.1f%%",
      sensor_msg.pipeline_heading_degrees,
      sensor_msg.current_milliamps,
      sensor_msg.depth_meters,
      sensor_msg.signal_strength_percent);
  }

  std::string image_topic_;
  std::string image_type_;
  std::string output_topic_;
  double inference_rate_hz_{2.0};
  std::string output_frame_id_;
  std::string api_base_url_;
  double api_timeout_sec_{5.0};
  bool debug_{false};
  std::string qos_reliability_;
  int qos_history_depth_{1};

  std::mutex image_mutex_;
  std_msgs::msg::Header latest_header_;
  cv::Mat latest_cv_image_;
  std::vector<uint8_t> latest_jpeg_bytes_;
  bool has_latest_frame_{false};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_subscription_;
  rclcpp::Publisher<screen_ocr_msgs::msg::SensorMsg>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace screen_ocr

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<screen_ocr::PipelineLocatorNode>());
  rclcpp::shutdown();
  return 0;
}
