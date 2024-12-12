/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(cluster/crush/delete,FixClusterCrushDelete);
// clang-format on
#else

#ifndef LAMMPS_FIX_CLUSTER_CRUSH_DELETE_H
#define LAMMPS_FIX_CLUSTER_CRUSH_DELETE_H

#include "fix.h"
#include "nucc_cspan.hpp"
#include <array>

enum class DIST {DIST_UNIFORM,DIST_GAUSSIAN};

namespace LAMMPS_NS {
class FixClusterCrushDelete : public Fix {
 public:
  FixClusterCrushDelete(class LAMMPS *lmp, int narg, char **arg);
  ~FixClusterCrushDelete() noexcept(true) override;
  void init() override;
  int setmask() override;
  void pre_exchange() override;

 protected:
  // necessary things for computation

  class Region *region = nullptr;
  class ComputeClusterSizeExt *compute_cluster_size = nullptr;
  class ComputeClusterTemp *compute_temp = nullptr;

  FILE *fp = nullptr;

  bigint next_step = 0;

  int nloc = 0;
  NUCC::cspan<int> p2m;
  NUCC::cspan<int> pproc;    // number of atoms to move per rank
  NUCC::cspan<int> c2c;
  std::array<double, 6> sbonds{};
  std::array<double, 6> vels{};
  std::array<double, 3> xmid{};
  int to_insert = 0;

  // parameters
  int screenflag = 0;
  int fileflag = 1;
  int scaleflag = 0;
  int kmax = 0;
  double overlap = 0;
  double overlapsq = 0;
  int maxtry = 1000;
  int ntype = 0;
  int groupid = 0;

  //velocity and coordinates
  bool fix_temp = 0;
  double monomer_temperature = 0;
  class RanPark* vrandom = nullptr;
  double vsigma = 0;
  DIST vdist = DIST::DIST_GAUSSIAN;
  class RanPark* xrandom = nullptr;
  double xsigma = 0;
  DIST xdist = DIST::DIST_UNIFORM;
  int varflag = 0;
  char *vstr{}, *xstr{}, *ystr{}, *zstr{};
  std::array<int, 4> vars{};

  void deleteAtoms(int atoms2move_local) noexcept(true);
  void postDelete() noexcept(true);
  void add();
  int vartest(double x, double y, double z);
  void generate_velocity(double *vnew);
  bool check_overlap(double* coord);
};

}    // namespace LAMMPS_NS

#endif
#endif
