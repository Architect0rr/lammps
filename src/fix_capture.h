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

#include "fix.h"
#include "compute.h"
#include "random_park.h"
#include "region.h"

#include <unordered_map>

namespace LAMMPS_NS {

class FixCapture : public Fix {
 public:
  FixCapture(class LAMMPS *, int, char **);
  ~FixCapture() override;
  int setmask() override;
  void init() override;
  void final_integrate() override;

 protected:
  Region *region = nullptr;
  RanPark *vrandom = nullptr;
  Compute* compute_temp = nullptr;

  std::unordered_map<int, std::pair<double, double>> typeids;

  int nsigma;
  FILE* logfile;

  double xlo, ylo, zlo, xhi, yhi, zhi;
  double lamda[3];
  double *boxlo, *boxhi;
  double xone[3];

};

}    // namespace LAMMPS_NS

#endif
#endif