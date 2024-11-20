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
ComputeStyle(cluster/pe,ComputeClusterPE);
// clang-format on
#else

#ifndef LMP_COMPUTE_CLUSTER_PE_H
#define LMP_COMPUTE_CLUSTER_PE_H

#include "compute.h"
#include "compute_cluster_size.h"

namespace LAMMPS_NS {

class ComputeClusterPE : public Compute {
 public:
  ComputeClusterPE(class LAMMPS *lmp, int narg, char **arg);
  ~ComputeClusterPE() noexcept(true) override;
  void init() override;
  void compute_vector() override;
  void compute_local() override;
  double memory_usage() override;

 private:
  ComputeClusterSize *compute_cluster_size = nullptr;
  Compute *compute_pe_atom = nullptr;

  double *pes = nullptr;          // array of pes of global clusters
  double *local_pes = nullptr;    // array of pes of local clusters
  int size_cutoff;                // size of max cluster
};

}    // namespace LAMMPS_NS

#endif
#endif
