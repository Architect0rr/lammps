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

#ifndef LAMMPS_FIX_CRUSH_H
#define LAMMPS_FIX_CRUSH_H

#include "fix.h"
#include "compute.h"
#include "region.h"
#include "random_park.h"

#include <map>
#include <set>
#include <vector>

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
  Compute *compute_cluster_atom = nullptr;

  RanPark *xrandom = nullptr;
  RanPark *vrandom = nullptr;

  std::map<tagint, std::vector<tagint>> atoms_by_cID; // Mapping cID  -> local idx
  std::map<tagint, std::vector<tagint>> cIDs_by_size; // Mapping size -> cIDs

  FILE* fp;
  int screenflag, fileflag;

  int maxtry, triclinic, scaleflag, fix_temp;
  int kmax;
  double monomer_temperature, odistsq;

  double xlo, ylo, zlo, xhi, yhi, zhi;
  double lamda[3], *coord;
  double *boxlo, *boxhi;
  double xone[3];

  bool gen_one();
  double maxwell_distribution3D(double, double, double) noexcept(true);
  long double erfinv(long double) noexcept(true);
  long double erfinv_refine(long double, int) noexcept(true);
};

} // namespace LAMMPS_NS

#endif
#endif
