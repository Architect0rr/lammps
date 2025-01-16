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
ComputeStyle(cluster/enthropy,ComputeClusterEnthropy);
// clang-format on
#else

#ifndef LMP_COMPUTE_CLUSTER_ENTHROPY_H
#define LMP_COMPUTE_CLUSTER_ENTHROPY_H

#include "compute.h"
#include "nucc_cspan.hpp"

namespace LAMMPS_NS {

class ComputeClusterEnthropy : public Compute {
 public:
  ComputeClusterEnthropy(class LAMMPS *lmp, int narg, char **arg);
  ~ComputeClusterEnthropy() noexcept(true) override;
  void init() override;
  void compute_vector() override;
  void compute_local() override;
  double memory_usage() override;

 private:
  class ComputeClusterSizeExt *compute_cluster_size = nullptr;
  Compute *compute_entropy_atom = nullptr;

  NUCC::cspan<double> enth;          // array of temps of global clusters
  NUCC::cspan<double> local_enth;    // array of temps of local clusters
  int size_cutoff;                 // size of max cluster
};

}    // namespace LAMMPS_NS

#endif
#endif
