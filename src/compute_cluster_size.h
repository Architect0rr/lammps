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

#ifndef LMP_COMPUTE_CLUSTER_SIZE_BASE_H
#define LMP_COMPUTE_CLUSTER_SIZE_BASE_H

#include "compute.h"
#    include "nucc_defs.hpp"

namespace LAMMPS_NS {
class ComputeClusterSize : public Compute {
 public:
  ComputeClusterSize(class LAMMPS *lmp, int narg, char **arg): Compute(lmp, narg, arg) {};

  inline constexpr virtual int get_size_cutoff() const noexcept = 0;
  inline constexpr virtual NUCC::cspan<const double> get_data() const noexcept = 0;

  inline constexpr virtual const NUCC::Map_t<int, NUCC::Vec_t<int>>* get_cIDs_by_size() const noexcept = 0;

  int is_avg;
};

}    // namespace LAMMPS_NS

#endif
