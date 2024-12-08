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

#  ifndef LMP_COMPUTE_CLUSTER_SIZE_ExT_H
#    define LMP_COMPUTE_CLUSTER_SIZE_ExT_H

#    define LMP_NUCC_ALLOC_COEFF 1.2
#    define LMP_NUCC_CLUSTER_MAX_OWNERS 128
#    define LMP_NUCC_CLUSTER_MAX_SIZE 300
#    define LMP_NUCC_CLUSTER_MAX_GHOST 300

#    include "compute.h"
#    include "nucc_allocator.hpp"
#    include "nucc_defs.hpp"
#    include "nucc_cspan.hpp"

#    include <array>
#    include <scoped_allocator>
#    include <span>
#    include <unordered_map>
#    include <vector>

namespace NUCC {

struct cluster_data {
  explicit cluster_data(const int _clid) : clid(_clid) {}

  void rearrange() noexcept { ::memcpy(_atoms + l_size, _ghost, nghost * sizeof(int)); }

  NUCC::cspan<const int> atoms_all() const { return std::span<const int>(_atoms, l_size + nghost); }

  template <bool Protect = true>
    requires(!Protect)
  NUCC::cspan<int, LMP_NUCC_CLUSTER_MAX_SIZE> atoms()
  {
    return std::span<int, LMP_NUCC_CLUSTER_MAX_SIZE>(_atoms, LMP_NUCC_CLUSTER_MAX_SIZE);
  }

  NUCC::cspan<const int> atoms() const { return std::span<const int>(_atoms, l_size); }

  NUCC::cspan<int, LMP_NUCC_CLUSTER_MAX_GHOST> ghost_initial() { return std::span<int, LMP_NUCC_CLUSTER_MAX_GHOST>(_ghost, nghost); }

  NUCC::cspan<const int> ghost() const { return std::span<const int>(_atoms + l_size, nghost); }

  template <bool Protect = true>
    requires(!Protect)
  NUCC::cspan<int, LMP_NUCC_CLUSTER_MAX_OWNERS> owners()
  {
    return std::span<int, LMP_NUCC_CLUSTER_MAX_OWNERS>(_owners, LMP_NUCC_CLUSTER_MAX_OWNERS);
  }

  NUCC::cspan<const int> owners() const { return std::span<const int>(_owners, nowners); }

  int clid = 0;       // cluster ID
  int l_size = 0;     // local size
  int g_size = 0;     // global size
  int host = -1;      // host proc (me if <0)
  int nhost = 0;      // local cluster size of host proc
  int nowners = 0;    // number of owners
  int nghost = 0;     // number of ghost atoms in cluster
 private:
  int _owners[LMP_NUCC_CLUSTER_MAX_OWNERS];    // procs owning some cluster's atoms
  int _atoms[LMP_NUCC_CLUSTER_MAX_SIZE];       // local ids of atoms
  int _ghost[LMP_NUCC_CLUSTER_MAX_GHOST];      // local ids of ghost atoms
};

}    // namespace NUCC

namespace LAMMPS_NS {

class ComputeClusterSizeExt : public Compute {
 public:
  ComputeClusterSizeExt(class LAMMPS *lmp, int narg, char **arg);
  ~ComputeClusterSizeExt() noexcept(true) override;
  void init() override;
  void compute_vector() override;
  void compute_peratom() override;
  double memory_usage() override;

  inline constexpr int get_size_cutoff() const noexcept(true) { return size_cutoff; }
  inline constexpr int get_nonexclusive() const noexcept(true) { return nonexclusive; }
  // inline constexpr const std::unordered_map<int, int> &get_cluster_map() const noexcept(true) { return cluster_map; }
  // inline constexpr const std::unordered_map<int, std::vector<int>> &get_cIDs_by_size() const noexcept(true) { return cIDs_by_size; }
  // inline constexpr const std::unordered_map<int, std::vector<int>> &get_cIDs_by_size_all() const noexcept(true) { return cIDs_by_size_all; }
  inline constexpr const NUCC::Map_t<int, int> *get_cluster_map() const noexcept(true) { return cluster_map; }
  inline constexpr const NUCC::Map_t<int, NUCC::Vec_t<int>> *get_cIDs_by_size() const noexcept(true) { return cIDs_by_size; }
  inline constexpr const NUCC::Map_t<int, NUCC::Vec_t<int>> *get_cIDs_by_size_all() const noexcept(true) { return cIDs_by_size_all; }
  inline constexpr const NUCC::cspan<const NUCC::cluster_data> get_clusters() const noexcept(true) { return clusters; }

 private:
  int size_cutoff;    // number of elements reserved in dist

  NUCC::MemoryKeeper *keeper1;
  NUCC::MapAlloc_t<int, int> *cluster_map_allocator;
  NUCC::Map_t<int, int> *cluster_map;
  // std::unordered_map<int, int> cluster_map;    // clid -> idx

  NUCC::MemoryKeeper *keeper2;
  NUCC::MapAlloc_t<int, NUCC::Vec_t<int>> *alloc_map_vec1;
  NUCC::Map_t<int, NUCC::Vec_t<int>> *cIDs_by_size;
  // std::unordered_map<int, std::vector<int>> cIDs_by_size;    // size -> vector(idx)

  NUCC::MemoryKeeper *keeper3;
  NUCC::MapAlloc_t<int, NUCC::Vec_t<int>> *alloc_map_vec2;
  NUCC::Map_t<int, NUCC::Vec_t<int>> *cIDs_by_size_all;
  // std::unordered_map<int, std::vector<int>> cIDs_by_size_all;

  int nloc;                          // number of reserved elements in atoms_by_cID and cIDs_by_size
  NUCC::cspan<double> dist;          // cluster size distribution (vector == dist)
  NUCC::cspan<double> dist_local;    // local cluster size distribution
  int nc_global;                     // number of clusters total
  NUCC::cspan<int> counts_global;
  NUCC::cspan<int> displs;
  NUCC::cspan<NUCC::cluster_data> clusters;
  NUCC::cspan<int> ns;
  NUCC::cspan<int> gathered;
  NUCC::cspan<double> peratom_size;
  bigint natom_loc;
  int nonexclusive;
  int nloc_peratom;

  NUCC::cspan<int> monomers;
  int nmono;

  Compute *compute_cluster_atom = nullptr;
  // void test_allocator() const;
};

}    // namespace LAMMPS_NS

#  endif
#endif
