#include <opencv2/opencv.hpp>

#include "include/img_tools.hpp"

int main()
{
  // 读取图片
  cv::Mat bgr_img;
  bgr_img = cv::imread("imgs/armor2.jpg");

  // 彩色图转灰度图
  cv::Mat gray_img;
  cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);
  // 显示灰度图, 调试用
  cv::resize(gray_img, gray_img, {}, 0.5, 0.5);
  cv::imshow("gray", gray_img);
  cv::resize(gray_img, gray_img, {}, 2, 2);

  // 进行二值化
  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, 120, 255, cv::THRESH_BINARY);
  // 显示二值图, 调试用
  cv::resize(binary_img, binary_img, {}, 0.5, 0.5);
  cv::imshow("binary_img", binary_img);
  cv::resize(binary_img, binary_img, {}, 2, 2);

  // 获取轮廓点
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  // 显示轮廓点, 调试用
  cv::Mat drawcontours = bgr_img.clone();
  for (const auto & contour : contours) {
    tools::draw_points(drawcontours, contour);
  }
  cv::resize(drawcontours, drawcontours, {}, 0.5, 0.5);
  cv::imshow("drawcontours", drawcontours);

  // 获取旋转矩形并显示, 示范用
  std::vector<cv::RotatedRect> rotated_rects;
  for (const auto & contour : contours) {
    auto rotated_rect = cv::minAreaRect(contour);
    rotated_rects.emplace_back(rotated_rect);
  }

  cv::Mat drawrect = bgr_img.clone();
  for (const auto & rotated_rect : rotated_rects) {
    std::vector<cv::Point2f> points(4);
    rotated_rect.points(&points[0]);
    tools::draw_points(drawrect, points);
  }
  cv::resize(drawrect, drawrect, {}, 0.5, 0.5);
  cv::imshow("drawrect", drawrect);

  cv::waitKey(0);
  return 0;
}
