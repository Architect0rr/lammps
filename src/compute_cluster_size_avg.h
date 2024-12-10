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
ComputeStyle(size/cluster/avg,ComputeClusterSize);
// clang-format on
#else

#ifndef LMP_COMPUTE_CLUSTER_SIZE_AVG_H
#define LMP_COMPUTE_CLUSTER_SIZE_AVG_H

#include "compute_cluster_size.h"

#    include "nucc_allocator.hpp"
#    include "nucc_defs.hpp"
#    include "nucc_cspan.hpp"
#include <unordered_map>

namespace LAMMPS_NS {
class ComputeClusterSizeAVG : public ComputeClusterSize {
 public:
  ComputeClusterSizeAVG(class LAMMPS *lmp, int narg, char **arg);
  ~ComputeClusterSizeAVG() noexcept(true) override;
  void init() override;
  void compute_vector() override;
  void compute_peratom() override;
  double memory_usage() override;

  inline constexpr int get_size_cutoff() const noexcept override { return size_cutoff; }
  inline constexpr NUCC::cspan<const double> get_data() const noexcept override { return dist; }

  inline constexpr const std::unordered_map<int, std::vector<int>>* get_atoms_by_cID() const noexcept { return &atoms_by_cID; }
  inline constexpr virtual const NUCC::Map_t<int, NUCC::Vec_t<int>>* get_cIDs_by_size() const noexcept override { return cIDs_by_size; }


 private:
  NUCC::MemoryKeeper *keeper;
  NUCC::MapAlloc_t<int, NUCC::Vec_t<int>> *alloc_map_vec;
  NUCC::Map_t<int, NUCC::Vec_t<int>> *cIDs_by_size;

  std::unordered_map<int, std::vector<int>> atoms_by_cID;    // Mapping cID  -> local idx
  // std::unordered_map<int, std::vector<int>> _cIDs_by_size;    // Mapping size -> cIDs

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
