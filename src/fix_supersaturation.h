/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(supersaturation,FixSupersaturation);
// clang-format on
#else

#ifndef LAMMPS_FIX_SUPERSATURATION_H
#define LAMMPS_FIX_SUPERSATURATION_H

#include <bits/types/struct_FILE.h>
#include <bits/types/struct_FILE.h>
#include "compute_supersaturation_mono.h"

#include "fix.h"
#include "random_park.h"
#include "region.h"

namespace LAMMPS_NS {

class FixSupersaturation : public Fix {
 public:
  FixSupersaturation(class LAMMPS *, int, char **);
  ~FixSupersaturation() override;
  int setmask() override;
  void init() override;
  void pre_exchange() override;

 protected:
  Region *region = nullptr;
  ComputeSupersaturationMono *compute_supersaturation_mono = nullptr;

  RanPark *xrandom = nullptr;
  RanPark *vrandom = nullptr;

  FILE *fp;
  int screenflag, fileflag;
  FILE log;

  bigint next_step;

  int maxtry, triclinic, scaleflag, fix_temp;
  double monomer_temperature, odistsq;
  double supersaturation{};

  double xlo, ylo, zlo, xhi, yhi, zhi;
  double lamda[3]{};
  double *boxlo, *boxhi;
  double xone[3]{};

  int *pproc{};
  int maxtry_call, ntype;

  void delete_monomers() noexcept(true);
  void add_monomers() noexcept(true);
  bool gen_one() noexcept(true);
};

}    // namespace LAMMPS_NS

#endif
#endif