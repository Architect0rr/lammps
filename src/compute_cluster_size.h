/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(cluster/size,ComputeClusterSize);
// clang-format on
#else

#ifndef LMP_COMPUTE_CLUSTER_SIZE_H
#define LMP_COMPUTE_CLUSTER_SIZE_H

#include "compute.h"
#include "region.h"

#include <unordered_map>

namespace LAMMPS_NS {

class ComputeClusterSize : public Compute {
 public:
  ComputeClusterSize(class LAMMPS *lmp, int narg, char **arg);
  ~ComputeClusterSize() noexcept(true) override;
  void init() override;
  void compute_vector() override;
  double memory_usage() override;

  std::unordered_map<tagint, std::vector<tagint>> atoms_by_cID;    // Mapping cID  -> local idx
  std::unordered_map<tagint, std::vector<tagint>> cIDs_by_size;    // Mapping size -> cIDs

 private:
  int nloc;            // number of reserved elements in atoms_by_cID and cIDs_by_size
  double *dist;        // cluster size distribution (vector == dist)
  bigint nc_global;    // number of clusters total
  int size_cutoff;     // number of elements reserved in dist

  Region *region = nullptr;
  Compute *compute_cluster_atom = nullptr;
};

}    // namespace LAMMPS_NS

#endif
#endif
