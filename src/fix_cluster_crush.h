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

enum class MODE { DELETE, TELEPORT, FASTPORT };

class FixClusterCrush : public Fix {
 public:
  FixClusterCrush(class LAMMPS *lmp, int narg, char **arg);
  ~FixClusterCrush() noexcept(true) override;
  void init() override;
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
  int triclinic;
  int scaleflag;

  bigint next_step;
  int kmax;

  int nloc;
  int *p2m;
  int *pproc;    // number of atoms to move per rank
  bigint *c2c;

  double globbonds[3][2]{};
  double subbonds[3][2]{};
  double *coord{};
  double lamda[3]{};
  double *boxlo;
  double *boxhi;
  double xone[3]{};

  MODE mode;

  // for both teleport and fastport
  bool fix_temp;
  double monomer_temperature;
  double odistsq;
  double overlap;

  // for teleport
  int velscaleflag;
  double velscale;
  int maxtry;

  // for fastport
  RanPark *algorand;
  int maxtry_call;
  int *map;
  bigint ncell[3]{};
  int ntype;
  double sigma;
  int *succ;

  bool genOneFull() noexcept(true);
  void set(int pID) noexcept(true);
  void set_speed(int pID) noexcept(true);
  void deleteAtoms(int atoms2move_local) noexcept(true);
  void postTeleport() noexcept(true);
  void postDelete() noexcept(true);
  void fill_map() noexcept(true);
  void add_core() noexcept(true);
  void build_tp_map() noexcept(true);
  void post_add(const int nlocal_previous) noexcept(true);
  inline bigint i2c(bigint i, bigint j, bigint k) const noexcept(true);
  inline bigint x2c(double x, double y, double z) const noexcept(true);
};

}    // namespace LAMMPS_NS

#endif
#endif
