#include "detector.hpp"

#include <opencv2/dnn.hpp>   // cv::dnn::Net, readNetFromONNX, blobFromImage
#include <opencv2/imgproc.hpp>
#include <algorithm>         // std::max_element

namespace auto_aim
{
  std::list<Armor> Detector::detect(const cv::Mat &bgr_img)
  {
    // TODO
    //
    // 1) bgr -> gray
    //
    // 2) gray -> binary (阈值化)
    //
    // 3) 查找轮廓
    //
    // 4) 获取灯条: 遍历轮廓 -> minAreaRect -> 构造 Lightbar -> 几何筛选 (可以使用lightbar.angle_error，ratio，length) -> 统计颜色
    //
    // 5) lightbars 由左到右排序, 两两配对：
    //    - 颜色相同
    //    - 构造 Armor
    //    - 几何筛选（可以使用armor.ratio，side_ratio，rectangular_error）
    //    - 裁剪/提取装甲板ROI存到 armor.pattern
    //    - 调用 classify(armor)
    //    - 名称&置信度筛选（armor.confidence，name)
    //    - 通过则 push 到 armors
    //
    // 7) return armors;

    (void)bgr_img; // 避免未使用参数警告（复现 detect 后可删除）
    return {};
  }

  void Detector::classify(Armor &armor)
  {
    // === 1) 加载 ONNX 分类网络（tiny_resnet） ===
    // 说明：当前实现每次调用都会读一次模型（很慢），但这里按“原有代码”保留该写法。
    cv::dnn::Net net = cv::dnn::readNetFromONNX("tiny_resnet.onnx");

    // === 2) 预处理：armor.pattern(BGR ROI) -> 灰度 ===
    cv::Mat gray;
    cv::cvtColor(armor.pattern, gray, cv::COLOR_BGR2GRAY);

    // === 3) 预处理：等比例缩放到 32x32 输入 ===
    // 做法：创建 32x32 的黑底画布 input，把缩放后的灰度图贴到左上角(0,0,w,h)。
    // （这相当于 letterbox，但没有居中）
    auto input = cv::Mat(32, 32, CV_8UC1, cv::Scalar(0));
    auto x_scale = static_cast<double>(32) / gray.cols;
    auto y_scale = static_cast<double>(32) / gray.rows;
    auto scale = std::min(x_scale, y_scale);
    auto h = static_cast<int>(gray.rows * scale);
    auto w = static_cast<int>(gray.cols * scale);
    auto roi = cv::Rect(0, 0, w, h);
    cv::resize(gray, input(roi), {w, h});

    // === 4) 组织成 DNN 的输入 blob，并归一化到 [0,1] ===
    auto blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar());

    // === 5) 前向推理 ===
    net.setInput(blob);
    cv::Mat outputs = net.forward(); // 一般是 (1, num_classes) 或类似形状的 logits

    // === 6) softmax：logits -> 概率（稳定版：先减 max） ===
    float max = *std::max_element(outputs.begin<float>(), outputs.end<float>());
    cv::exp(outputs - max, outputs);
    float sum = cv::sum(outputs)[0];
    outputs /= sum;

    // === 7) argmax：取最大概率的类别作为预测结果 ===
    double confidence;
    cv::Point label_point;
    cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
    int label_id = label_point.x;

    // === 8) 写回 Armor：置信度 + 类别 ===
    armor.confidence = confidence;
    armor.name = static_cast<ArmorName>(label_id);
  }

} // namespace auto_aim