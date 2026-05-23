#ifndef AUTO_AIM__DETECTOR_HPP
#define AUTO_AIM__DETECTOR_HPP

#include <list>
#include <opencv2/opencv.hpp>

#include "armor.hpp"

namespace auto_aim
{
class Detector
{
public:
  std::list<Armor> detect(const cv::Mat & bgr_img);

private:
  void classify(Armor & armor);
};

}  // namespace auto_aim

#endif
