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
ComputeStyle(cluster/size/ext,ComputeClusterSizeExt);
// clang-format on
#else

#ifndef LMP_COMPUTE_CLUSTER_SIZE_ExT_H
#define LMP_COMPUTE_CLUSTER_SIZE_ExT_H

#define LMP_NUCC_ALLOC_COEFF 1.2
#define LMP_NUCC_CLUSTER_MAX_OWNERS 128
#define LMP_NUCC_CLUSTER_MAX_SIZE 300
#define LMP_NUCC_CLUSTER_MAX_GHOST 30
#include "compute.h"

#include <unordered_map>

namespace LAMMPS_NS {

class ComputeClusterVolume;

struct cluster_data {
  int l_size = 0;     // local size
  int g_size = 0;     // global size
  int host = -1;      // host proc (me if <0)
  int nhost = 0;      // local cluster size of host proc
  int nowners = 0;    // number of owners
  int nghost = 0;     // number of ghost atoms in cluster
  int owners[LMP_NUCC_CLUSTER_MAX_OWNERS];
  int atoms[LMP_NUCC_CLUSTER_MAX_SIZE];     // local ids of atoms
  int ghost[LMP_NUCC_CLUSTER_MAX_GHOST];    // local ids of ghost atoms
};

class ComputeClusterSizeExt : public Compute {
 public:
  friend class ComputeClusterVolume;
  ComputeClusterSizeExt(class LAMMPS *lmp, int narg, char **arg);
  ~ComputeClusterSizeExt() noexcept(true) override;
  void init() override;
  void compute_vector() override;
  double memory_usage() override;

  inline int get_size_cutoff() const noexcept(true) { return size_cutoff; }

 private:
  int size_cutoff;    // number of elements reserved in dist

  // MemoryKeeper* keeper;
  // CustomAllocator<std::pair<const int, cluster_ptr>>* alloc;
  // std::unordered_map<int, cluster_ptr, std::hash<int>, std::equal_to<int>, CustomAllocator<std::pair<const int, cluster_ptr>>>* cluster_map;
  std::unordered_map<int, int> cluster_map;

  int nloc;                // number of reserved elements in atoms_by_cID and cIDs_by_size
  double *dist;            // cluster size distribution (vector == dist)
  double *dist_local{};    // local cluster size distribution
  int nc_global;           // number of clusters total
  int *counts_global{};
  int *displs{};
  cluster_data *clusters{};
  int *ns{};
  int *gathered{};
  int natom_loc;

  Compute *compute_cluster_atom = nullptr;
  // void test_allocator() const;
};

}    // namespace LAMMPS_NS

#endif
#endif
