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
  double *kes = nullptr;          // array of kes of global clusters
  double *local_kes = nullptr;    // array of kes of local clusters
  int size_cutoff;                // size of max cluster
  double occupied_volume(const double **const centers, const int *const list, const int n,
                         const double r, const double voxel_size) const;
  double occupied_volume2(const double **const centers, const int *const list, const int n,
                          const double r, const double voxel_size) const;
};

}    // namespace LAMMPS_NS

#endif
#endif
