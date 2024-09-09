/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(supersaturation/volume,FixSupersaturationVolume);
// clang-format on
#else

#ifndef LAMMPS_FIX_SUPERSATURATION_VOLUME_H
#define LAMMPS_FIX_SUPERSATURATION_VOLUME_H

#include "compute_supersaturation_mono.h"
#include "neigh_list.h"

#include "fix.h"

namespace LAMMPS_NS {

class FixSupersaturationVolume : public Fix {
 public:
  FixSupersaturationVolume(class LAMMPS *, int, char **);
  ~FixSupersaturationVolume() noexcept(true) override;
  int setmask() override;
  void init() override;
  void init_list(int, class NeighList *) override;
//   void end_of_step() override;
  void pre_exchange() override;

 protected:
  ComputeSupersaturationMono *compute_supersaturation_mono = nullptr;

  NeighList* list;

  bool need_exchange;

  FILE *fp;
  int screenflag;
  int fileflag;

  double supersaturation;
  double damp;

  int offflag;
  int start_offset;

  bigint next_step;

  std::vector<Fix *> rfix;

  void remap_before() noexcept(true);
  void remap_after() noexcept(true);
  void print_box();
  void calculate_out();
};

}    // namespace LAMMPS_NS

#endif
#endif