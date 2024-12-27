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

// TODO: NUCC FILE

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(cluster/volume,ComputeClusterVolume);
// clang-format on
#else

#ifndef LMP_COMPUTE_CLUSTER_VOLUME_H
#define LMP_COMPUTE_CLUSTER_VOLUME_H

#include "compute.h"
#include "nucc_cspan.hpp"
#include <array>

namespace LAMMPS_NS {

enum class VOLUMEMODE { RECTANGLE = 0, SPHERE = 1, CALC = 2 };

class ComputeClusterSizeExt;

class ComputeClusterVolume : public Compute {
 public:
  ComputeClusterVolume(class LAMMPS *lmp, int narg, char **arg);
  ~ComputeClusterVolume() noexcept(true) override;
  void init() override;
  void compute_vector() override;
  void compute_local() override;
  double memory_usage() override;

 private:
  ComputeClusterSizeExt *compute_cluster_size = nullptr;

  VOLUMEMODE mode;
  std::array<double, 6> subbonds;
  int nloc{};
  NUCC::cspan<double> volumes;
  NUCC::cspan<double> dist_local;
  NUCC::cspan<double> dist;
  int size_cutoff;    // size of max cluster
  double voxel_size{};
  double overlap{};
  double overlap_sq{};

  // VOLUMEMODE::CALC
  NUCC::cspan<bool> occupancy_grid;
  bigint nloc_grid;
  int n_cells;
  bool precompute;
  NUCC::cspan<int> offsets;
  int noff;
  int noffsets{};

  // VOLUMEMODE::RECTANGLE
  NUCC::cspan<double> bboxes;

  // MPI_COMMUNICATION
  ::MPI_Request *in_reqs{};
  ::MPI_Request *out_reqs{};

  //   ::MPI_Status *in_stats;
  //   ::MPI_Status *out_stats;

  NUCC::cspan<double> recv_buf;
  bigint nloc_recv;
  NUCC::cspan<double> dist_global;

  // int *recv_comm_matrix_local;
  // int *recv_comm_matrix_global;
  // int *send_comm_matrix_local;
  // int *send_comm_matrix_global;

  // with AVX512 occupied_volume_grid works faster, than precomputed version
  double occupied_volume_precomputed(const NUCC::cspan<const int> &list, const int n,
                                     const int nghost, const bool nonexclusive) noexcept;
  double occupied_volume_grid(const int *const list, const int n, const int nghost,
                              const bool nonexclusive) noexcept;
  void cluster_bbox(const NUCC::cspan<const int> &list, const int n, double *bbox) const noexcept;

  template <typename TYPE> TYPE **create_ptr_array(TYPE **&ptr, int n, const char *name)
  {
    ptr = n <= 0 ? nullptr : static_cast<TYPE **>(memory->smalloc(sizeof(TYPE *) * n, name));
    // for (int i = 0; i < n; ++i) { array[i] = nullptr; }
    return ptr;
  }

  template <typename TYPE> TYPE **grow_ptr_array(TYPE **&ptr, int n, const char *name)
  {
    if (n <= 0) {
      memory->destroy(ptr);
      return nullptr;
    }

    if (ptr == nullptr) return create_ptr_array(ptr, n, name);

    ptr = static_cast<TYPE **>(memory->srealloc(ptr, sizeof(TYPE *) * n, name));
    return ptr;
  }
};

}    // namespace LAMMPS_NS

#endif
#endif
