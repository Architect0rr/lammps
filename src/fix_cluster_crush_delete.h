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

#include "compute.h"
#include "compute_cluster_size.h"
#include "fix.h"
#include "fix_regen.h"
#include "region.h"

namespace LAMMPS_NS {

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
  Compute *compute_temp = nullptr;
  FixRegen *fix_regen = nullptr;

  int xseed;

  FILE *fp;
  int screenflag;
  int fileflag;
  int scaleflag;

  bigint next_step;
  int kmax;

  int nloc;
  int *p2m;
  int *pproc{};    // number of atoms to move per rank
  bigint *c2c{};

  int at_once;
  bool fix_temp;
  double monomer_temperature;
  double overlap;
  int maxtry;
  int ntype;
  double sigma;
  bool reneigh_forced;
  bigint ninserted_prev;
  bigint ninsert_prev;

  void deleteAtoms(int atoms2move_local) noexcept(true);
  void postDelete() noexcept(true);
};

}    // namespace LAMMPS_NS

#endif
#endif
