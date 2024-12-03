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
#include "nucc_allocator.hpp"
#include <scoped_allocator>
#include <unordered_map>
#include <vector>

namespace LAMMPS_NS {

class ComputeClusterVolume;
using Cluster_map_t =
    std::unordered_map<int, int, std::hash<int>, std::equal_to<int>,
                       std::scoped_allocator_adaptor<CustomAllocator<std::pair<const int, int>>>>;
using Allocator_map_vector =
    CustomAllocator<std::pair<const int, std::vector<int, CustomAllocator<int>>>>;
using Sizes_map_t =
    std::unordered_map<int, std::vector<int, CustomAllocator<int>>, std::hash<int>,
                       std::equal_to<int>, std::scoped_allocator_adaptor<Allocator_map_vector>>;

struct cluster_data {
  cluster_data(const bigint clid = 0) : clid(clid) {}

  bigint clid = 0;                            // cluster ID
  int l_size = 0;                             // local size
  int g_size = 0;                             // global size
  int host = -1;                              // host proc (me if <0)
  int nhost = 0;                              // local cluster size of host proc
  int nowners = 0;                            // number of owners
  int nghost = 0;                             // number of ghost atoms in cluster
  int owners[LMP_NUCC_CLUSTER_MAX_OWNERS];    // procs owning some cluster's atoms
  int atoms[LMP_NUCC_CLUSTER_MAX_SIZE];       // local ids of atoms
  int ghost[LMP_NUCC_CLUSTER_MAX_GHOST];      // local ids of ghost atoms
};

class ComputeClusterSizeExt : public Compute {
 public:
  friend class ComputeClusterVolume;
  ComputeClusterSizeExt(class LAMMPS *lmp, int narg, char **arg);
  ~ComputeClusterSizeExt() noexcept(true) override;
  void init() override;
  void compute_vector() override;
  double memory_usage() override;

  int get_size_cutoff() const noexcept(true) { return size_cutoff; }

 private:
  int size_cutoff;    // number of elements reserved in dist

  MemoryKeeper *keeper;
  CustomAllocator<std::pair<const int, int>> *alloc;
  Cluster_map_t *cluster_map;
  // std::unordered_map<bigint, int> cluster_map; // clid -> idx
  Allocator_map_vector *alloc_vector;
  Sizes_map_t *cIDs_by_size;
  // std::unordered_map<int, std::vector<int>> cIDs_by_size; // size -> vector(idx)

  int nloc;                // number of reserved elements in atoms_by_cID and cIDs_by_size
  double *dist;            // cluster size distribution (vector == dist)
  double *dist_local{};    // local cluster size distribution
  int nc_global;           // number of clusters total
  int *counts_global{};
  int *displs{};
  cluster_data *clusters{};
  int *ns{};
  int *gathered{};
  bigint natom_loc;
  int nonexclusive;

  int *monomers;
  int nmono;

  Compute *compute_cluster_atom = nullptr;
  // void test_allocator() const;
};

}    // namespace LAMMPS_NS

#endif
#endif
