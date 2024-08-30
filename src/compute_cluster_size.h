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

// typedef NucC::Alloc<std::pair<const tagint, std::vector<tagint>>> myalloc;
// typedef std::unordered_map<tagint, std::vector<tagint>, std::hash<tagint>, std::less<tagint>, myalloc> mymap;

namespace LAMMPS_NS {

class ComputeClusterSize : public Compute {
 public:
  ComputeClusterSize(class LAMMPS *, int, char **);
  ~ComputeClusterSize() override;
  void init() override;
  void compute_vector() override;
  double memory_usage() override;

  // Mapping cID  -> local idx
  std::unordered_map<tagint, std::vector<tagint>> atoms_by_cID;
  // Mapping size -> cIDs
  std::unordered_map<tagint, std::vector<tagint>> cIDs_by_size;
  // std::unordered_set<tagint> unique_cIDs;

 private:
  double xlo{}, ylo{}, zlo{}, xhi{}, yhi{}, zhi{};
  double lamda[3]{};
  double *boxlo{}, *boxhi{};
  double sublo[3]{}, subhi[3]{};    // epsilon-extended proc sub-box for adding atoms

  //   myalloc alloc;
  int nloc;
  double *dist;
  bigint nc_global;

  Region *region = nullptr;
  Compute *compute_cluster_atom = nullptr;
};

}    // namespace LAMMPS_NS

#endif
#endif