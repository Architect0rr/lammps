/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

// TODO: NUCC FILE

#ifdef FIX_CLASS
// clang-format off
FixStyle(capture/vel,FixCaptureVel);
// clang-format on
#else

#ifndef LAMMPS_FIX_CAPTURE_VEL_H
#define LAMMPS_FIX_CAPTURE_VEL_H

#include "fix.h"
#include <vector>
#include <unordered_map>

namespace LAMMPS_NS {

class FixCaptureVel : public Fix {
 public:
  FixCaptureVel(class LAMMPS *lmp, int narg, char **arg);
  ~FixCaptureVel() noexcept(true) override;
  int setmask() override;
  void init() override;
  void initial_integrate(int /*vflag*/) override;
  void end_of_step() override;
  void pre_force(int /*vflag*/) override;
  void pre_exchange() override;
  void init_list(int /*id*/, NeighList *ptr) override;

 protected:
  class Region *region = nullptr;
  class RanPark *vrandom = nullptr;
  class Compute *compute_temp = nullptr;

  double *sigmas{};
  double **rmins{};
  double *vmax_coeffs{};

  bool screenflag = false;    // wether to print info to screen or not
  bool fileflag = true;      // wether to output info to file or not
  bool delete_overlap = false;

  int nsigma = 0;    // number of gaussian sigmas of spread to allow
  int nsigmasq = 0;
  FILE *fp = nullptr;      // logfile

  class NeighList* list = nullptr;
  std::vector<int> to_delete;

  bigint ncaptured[2]{};
  bigint ncaptured_global[2]{};

  std::unordered_map<bigint, bool> flags;

  void post_delete() noexcept(true);
  long double rminsq(const int i, const int j) noexcept(true);
};

}    // namespace LAMMPS_NS

#endif
#endif
