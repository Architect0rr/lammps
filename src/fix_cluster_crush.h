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

#include "compute_cluster_size.h"

#include "fix.h"
#include "random_park.h"
#include "region.h"

namespace LAMMPS_NS {

class FixClusterCrush : public Fix {
 public:
  FixClusterCrush(class LAMMPS *, int, char **);
  ~FixClusterCrush() override;
  int setmask() override;
  void init() override;
  void pre_exchange() override;

 protected:
  Region *region = nullptr;
  ComputeClusterSize *compute_cluster_size = nullptr;

  RanPark *xrandom = nullptr;
  RanPark *vrandom = nullptr;

  FILE *fp;
  int screenflag, fileflag;

  bigint next_step;
  int nevery;

  int maxtry, triclinic, scaleflag, fix_temp;
  int kmax;
  double monomer_temperature, odistsq;

  double xlo, ylo, zlo, xhi, yhi, zhi;
  double lamda[3];
  double *boxlo, *boxhi;
  double xone[3];

  bool gen_one();
};

}    // namespace LAMMPS_NS

#endif
#endif