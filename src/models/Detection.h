#pragma once

#include <string>
#include <tuple>

struct Landmarks {
  double xPos;
  double yPos;
  std::string name;
};

struct BoundingBox {
  double top;
  double bottom;
  double left;
  double right;

  std::tuple<double, double> TopLeft() { return std::make_tuple(top, left); }

  std::tuple<double, double> BottomRight() {
    return std::make_tuple(bottom, right);
  }

  double Width() { return std::abs(left - right); }

  double Height() { return std::abs(top - bottom); }
};
