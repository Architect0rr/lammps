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
#include "fix.h"
#include "memory.h"

namespace LAMMPS_NS {

class FixClusterDump : public Fix {
 public:
  FixClusterDump(class LAMMPS *lmp, int narg, char **arg);
  ~FixClusterDump() noexcept(true) override;
  int setmask() override;
  void init() override;
  void end_of_step() override;

 protected:
  Compute **compute_vectors{};
  Compute **compute_scalars{};
  int num_vectors;
  int num_scalars;

  FILE **file_vectors;
  FILE *file_scalars;

  int size_cutoff;

  template <typename TYPE> inline TYPE **create_ptr_array(TYPE **&array, int n, const char *name)
  {
    array = n <= 0 ? nullptr : static_cast<TYPE **>(memory->smalloc(sizeof(TYPE *) * n, name));
    return array;
  }
};

}    // namespace LAMMPS_NS

#endif
#endif
