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

#include "compute_supersaturation_mono.h"

#include "fix.h"
#include "random_park.h"
#include "region.h"


namespace LAMMPS_NS {

class FixSupersaturation : public Fix {
 public:
  FixSupersaturation(class LAMMPS *, int, char **);
  ~FixSupersaturation() noexcept(true) override;
  int setmask() override;
  void pre_exchange() override;

 protected:
  Region *region = nullptr;
  ComputeSupersaturationMono *compute_supersaturation_mono = nullptr;

  RanPark *xrandom;
  RanPark *vrandom;

  FILE *fp;
  int screenflag;
  int fileflag;

  bigint next_step;

  int maxtry;
  int triclinic;
  int scaleflag;
  int fix_temp;
  double monomer_temperature;
  double odistsq;
  double overlap;
  double supersaturation;
  double damp;
  int offflag;
  int start_offset;

  double xlo;
  double ylo;
  double zlo;
  double xhi;
  double yhi;
  double zhi;
  double lamda[3]{};
  double *boxlo;
  double *boxhi;
  double xone[3]{};

  int *pproc{};
  int maxtry_call;
  int ntype;

  void delete_monomers() noexcept(true);
  void add_monomers() noexcept(true);
  void add_monomers2() noexcept(true);
  bool gen_one() noexcept(true);
  bool gen_one(double, double, double, double, double, double) noexcept(true);
  void set_speed() noexcept(true);
  void post_add(const int) noexcept(true);
  void post_delete() noexcept(true);
};

}    // namespace LAMMPS_NS

#endif
#endif