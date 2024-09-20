/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(cluster/dump,FixClusterDump);
// clang-format on
#else

#ifndef LAMMPS_FIX_CLUSTER_DUMP_H
#define LAMMPS_FIX_CLUSTER_DUMP_H

#include "compute.h"
#include "compute_cluster_size.h"
#include "compute_cluster_temps.h"
#include "compute_supersaturation_mono.h"
#include "compute_supersaturation_density.h"
#include "fix.h"

namespace LAMMPS_NS {

class FixClusterDump : public Fix {
 public:
  FixClusterDump(class LAMMPS *lmp, int narg, char **arg);
  ~FixClusterDump() noexcept(true) override;
  int setmask() override;
  void init() override;
  void end_of_step() override;

 protected:
  Compute *compute_temp = nullptr;
  ComputeClusterSize *compute_cluster_size = nullptr;
  ComputeClusterTemp *compute_cluster_temp = nullptr;
  ComputeSupersaturationMono *compute_supersaturation_mono = nullptr;
  ComputeSupersaturationDensity *compute_supersaturation_density = nullptr;

  FILE *cldist;
  FILE *cltemp;
  FILE *scalars;

  bigint next_step;

  int size_cutoff;
};

}    // namespace LAMMPS_NS

#endif
#endif
