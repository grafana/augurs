#ifndef STAN_MATH_PRIM_SCAL_FUN_FMA_HPP
#define STAN_MATH_PRIM_SCAL_FUN_FMA_HPP

namespace stan {
namespace math {

inline double fma(double x, double y, double z) {
  return x*y + z;
}

}  // namespace math
}  // namespace stan
#endif
