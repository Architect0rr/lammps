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

#include "nucc_cspan.hpp"
#include <unordered_map>

namespace LAMMPS_NS {
class ComputeClusterSize : public Compute {
 public:
  ComputeClusterSize(class LAMMPS *lmp, int narg, char **arg);
  ~ComputeClusterSize() noexcept(true) override;
  void init() override;
  void compute_vector() override;
  void compute_peratom() override;
  double memory_usage() override;

  inline constexpr int get_size_cutoff() const noexcept { return size_cutoff; }
  inline constexpr NUCC::cspan<const double> get_data() const noexcept { return dist; }

  std::unordered_map<int, std::vector<int>> atoms_by_cID;    // Mapping cID  -> local idx
  std::unordered_map<int, std::vector<int>> cIDs_by_size;    // Mapping size -> cIDs

 private:
  int nloc;         // number of reserved elements in atoms_by_cID and cIDs_by_size
  int nloc_atom;    // nunber of reserved elements in peratom array
  NUCC::cspan<double> peratom_size;    // peratom array (size of cluster it is in)
  NUCC::cspan<double> dist;            // cluster size distribution (vector == dist)
  int nc_global;                       // number of clusters total
  int size_cutoff;                     // number of elements reserved in dist

  Compute *compute_cluster_atom = nullptr;
};

}    // namespace LAMMPS_NS

#endif
#endif
