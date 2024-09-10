/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(capture,FixCapture);
// clang-format on
#else

#ifndef LAMMPS_FIX_CAPTURE_H
#define LAMMPS_FIX_CAPTURE_H

#include "compute.h"
#include "fix.h"
#include "random_park.h"
#include "region.h"
#include <utility>

#include <unordered_map>

namespace LAMMPS_NS {

enum class ACTION { COUNT, SLOW, DELETE };

class FixCapture : public Fix {
 public:
  FixCapture(class LAMMPS *lmp, int narg, char **arg);
  ~FixCapture() noexcept(true) override;
  int setmask() override;
  void init() override;
  void final_integrate() override;

 protected:
  Region *region = nullptr;
  RanPark *vrandom = nullptr;
  Compute *compute_temp = nullptr;

  std::unordered_map<int, std::pair<double, double>> typeids;    // mapping type->(vmean,sigma)

  ACTION action;    // action to do on captured atoms

  bool screenflag;    // wether to print info to screen or not
  bool fileflag;      // wether to output info to file or not

  int nsigma;    // number of gaussian sigmas of spread to allow
  FILE *fp;      // logfile

  void post_delete() noexcept(true);
  void check_overlap() noexcept(true);
};

}    // namespace LAMMPS_NS

#endif
#endif
