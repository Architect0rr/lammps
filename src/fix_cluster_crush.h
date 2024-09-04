/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(cluster/crush,FixClusterCrush);
// clang-format on
#else

#ifndef LAMMPS_FIX_CLUSTER_CRUSH_H
#define LAMMPS_FIX_CLUSTER_CRUSH_H

#include "compute.h"
#include "compute_cluster_size.h"

#include "fix.h"
#include "random_park.h"
#include "region.h"

namespace LAMMPS_NS {

class FixClusterCrush : public Fix {
 public:
  FixClusterCrush(class LAMMPS *, int, char **);
  ~FixClusterCrush() noexcept(true) override;
  int setmask() override;
  void pre_exchange() override;

 protected:
  Region *region = nullptr;
  ComputeClusterSize *compute_cluster_size = nullptr;
  Compute *compute_temp = nullptr;

  RanPark *xrandom = nullptr;
  RanPark *vrandom = nullptr;

  FILE *fp;
  int screenflag;
  int fileflag;
  int velscaleflag;
  double velscale;

  bigint next_step;

  int maxtry;
  int triclinic;
  int scaleflag;
  int fix_temp;
  int kmax;
  double monomer_temperature;
  double odistsq;

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

  int nprocs;
  int *nptt_rank;    // number of atoms to move per rank
  bigint *c2c;
  int nloc;
  int *p2m;

  int teleportflag;

  bool gen_one() noexcept(true);
  void set(int) noexcept(true);
  void delete_monomers(int) noexcept(true);
  void post_teleport() noexcept(true);
  void post_delete() noexcept(true);
};

}    // namespace LAMMPS_NS

#endif
#endif