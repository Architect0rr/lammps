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
#include "fix_store_atom.h"
#include "random_park.h"
#include "region.h"

namespace LAMMPS_NS {

#define FIX_CAPTURE_FLAGS_COUNT 6
#define FIX_CAPTURE_TOTAL_FLAG 0
#define FIX_CAPTURE_vNaN_FLAG 1
#define FIX_CAPTURE_xNaN_FLAG 2
#define FIX_CAPTURE_OVERSPEED_FLAG 3
#define FIX_CAPTURE_OVERSPEED_REL_FLAG 4
#define FIX_CAPTURE_OVERLAP_FLAG 5

enum class ACTION { COUNT, SLOW, DELETE };

class FixCapture : public Fix {
 public:
  FixCapture(class LAMMPS *lmp, int narg, char **arg);
  ~FixCapture() noexcept(true) override;
  int setmask() override;
  void init() override;
  void pre_exchange() override;

 protected:
  Region *region = nullptr;
  RanPark *vrandom = nullptr;
  Compute *compute_temp = nullptr;

  double *sigmas{};
  double **rmins{};
  double *vmax_coeffs{};

  ACTION action;    // action to do on captured atoms

  bool screenflag;    // wether to print info to screen or not
  bool fileflag;      // wether to output info to file or not

  int nsigma;    // number of gaussian sigmas of spread to allow
  FILE *fp;      // logfile

  double vmean[3]{};
  bool allow;
  bigint Fcounts[FIX_CAPTURE_FLAGS_COUNT]{};

  void post_delete() noexcept(true);
  void captured() noexcept(true);
  void test_overlap(int i) noexcept(true);
  void test_xnonnum(int i) noexcept(true);
  void test_vnonnum(int i) noexcept(true);
  static bool isnonnumeric(const double *const vec3) noexcept(true);
  long double rmin(const int i, const int j) noexcept(true);

  template <bool ISREL>
  bool super_cond(const double *const v, const double *const vmeanx, const double sigma) const
      noexcept(true);
  template <bool ISREL> void test_superspeed(int *atomid) noexcept(true);
};

}    // namespace LAMMPS_NS

#endif
#endif
