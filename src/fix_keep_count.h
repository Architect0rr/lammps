/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(keep/count,FixKeepCount);
// clang-format on
#else

#ifndef LAMMPS_FIX_KEEP_COUNT_H
#define LAMMPS_FIX_KEEP_COUNT_H

#include "compute_supersaturation_mono.h"

#include "fix.h"
#include "random_park.h"
#include "region.h"

enum class MODE { LOCAL, UNIVERSE };

namespace LAMMPS_NS {

class FixKeepCount : public Fix {
 public:
  FixKeepCount(class LAMMPS *, int, char **);
  ~FixKeepCount() noexcept(true) override;
  int setmask() override;
  void pre_exchange() override;

 protected:
  Region *region = nullptr;
  ComputeSupersaturationMono *compute_supersaturation_mono = nullptr;

  RanPark *xrandom;
  RanPark *vrandom;
  RanPark *alogrand;

  FILE *fp;
  int screenflag;
  int fileflag;

  bigint next_step;
  bigint total_count;

  MODE mode;

  int maxtry;
  int triclinic;
  int scaleflag;
  int fix_temp;
  double monomer_temperature;
  double odistsq;
  double overlap;
  double damp;
  int offflag;
  int start_offset;

  double globbonds[3][2]{};
  double subbonds[3][2]{};
  double lamda[3]{};
  double *boxlo;
  double *boxhi;
  double *coord{};
  double xone[3]{};

  int *pproc{};
  int maxtry_call;
  int ntype;

  void delete_monomers() noexcept(true);
  void add_monomers() noexcept(true);
  void add_monomers2() noexcept(true);
  bool gen_one_sub() noexcept(true);
  bool gen_one_full() noexcept(true);
  bool gen_one_sub_at(double, double, double, double, double, double) noexcept(true);
  void set_speed(int) noexcept(true);
  void post_add(const int) noexcept(true);
  void post_delete() noexcept(true);
};

}    // namespace LAMMPS_NS

#endif
#endif