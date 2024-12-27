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
ComputeStyle(usage/ram,ComputeRAMUsage);
// clang-format on
#else

#ifndef LMP_COMPUTE_RAM_USAGE_H
#define LMP_COMPUTE_RAM_USAGE_H

#include "compute.h"
#include "nucc_cspan.hpp"
#include <array>

namespace LAMMPS_NS {

class ComputeClusterSizeExt;

class ComputeRAMUsage : public Compute {
 public:
  ComputeRAMUsage(class LAMMPS *lmp, int narg, char **arg);
  ~ComputeRAMUsage() noexcept override;
  double compute_scalar() override;
  void init() override;
  void compute_local() override;
  double memory_usage() override;

 private:
  uint64_t local_usage = 0;
  uint64_t getCurrentMemoryUsage();
};

}    // namespace LAMMPS_NS

#endif
#endif
