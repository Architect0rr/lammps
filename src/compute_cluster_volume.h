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
ComputeStyle(cluster/volume,ComputeClusterVolume);
// clang-format on
#else

#ifndef LMP_COMPUTE_CLUSTER_VOLUME_H
#define LMP_COMPUTE_CLUSTER_VOLUME_H

#include "compute.h"
#include "compute_cluster_size_ext.h"

namespace LAMMPS_NS {

enum class VOLUMEMODE { RECTANGLE = 0, SPHERE = 1, CALC = 2 };

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
  double subbonds[6]{};
  int nloc{};
  double *volumes{};
  double *dist_local{};
  double *dist{};
  int size_cutoff;    // size of max cluster
  double voxel_size{};
  double overlap{};
  double overlap_sq{};

  // VOLUMEMODE::CALC
  bool *occupancy_grid{};
  bigint nloc_grid;
  int n_cells;
  bool precompute;
  int *offsets{};
  int noff;
  int noffsets{};

  // VOLUMEMODE::RECTANGLE
  double *bboxes{};

  ::MPI_Request *in_reqs{};
  ::MPI_Request *out_reqs{};

  //   ::MPI_Status *in_stats;
  //   ::MPI_Status *out_stats;

  double *recv_buf{};
  bigint nloc_recv;

  // with AVX512 occupied_volume_grid works faster, than precomputed version
  double occupied_volume_precomputed(const int *const list, const int n, const int nghost,
                                     bool nonexclusive) noexcept;
  double occupied_volume_grid(const int *const list, const int n, const int nghost,
                              bool nonexclusive) noexcept;
  void cluster_bbox(const int *const list, const int n, double *bbox) const noexcept;
};

}    // namespace LAMMPS_NS

#endif
#endif
