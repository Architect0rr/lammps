/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(cluster/crush_delete,FixClusterCrushDelete);
// clang-format on
#else

#ifndef LAMMPS_FIX_CLUSTER_CRUSH_DELETE_H
#define LAMMPS_FIX_CLUSTER_CRUSH_DELETE_H

#include "fix.h"
#include "nucc_cspan.hpp"

namespace LAMMPS_NS {
class Region;
class ComputeClusterSize;
class ComputeClusterTemp;
class FixRegen;
class FixClusterCrushDelete : public Fix {
 public:
  FixClusterCrushDelete(class LAMMPS *lmp, int narg, char **arg);
  ~FixClusterCrushDelete() noexcept(true) override;
  void init() override;
  int setmask() override;
  void pre_exchange() override;

 protected:
  Region *region = nullptr;
  ComputeClusterSize *compute_cluster_size = nullptr;
  ComputeClusterTemp *compute_temp = nullptr;
  FixRegen *fix_regen = nullptr;

  int xseed;

  FILE *fp;
  int screenflag;
  int fileflag;
  int scaleflag;

  bigint next_step;
  int kmax;

  int nloc;
  NUCC::cspan<int> p2m;
  NUCC::cspan<int> pproc;    // number of atoms to move per rank
  NUCC::cspan<int> c2c;

  int at_once;
  std::string groupname;
  bool fix_temp;
  double monomer_temperature;
  double overlap;
  int maxtry;
  int ntype;
  double sigma;
  bool reneigh_forced;
  bigint ninserted_prev;

  void deleteAtoms(int atoms2move_local) noexcept(true);
  void postDelete() noexcept(true);
};

}    // namespace LAMMPS_NS

#endif
#endif
