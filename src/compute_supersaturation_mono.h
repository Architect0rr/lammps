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
ComputeStyle(supersaturation/mono,ComputeSupersaturationMono);
// clang-format on
#else

#ifndef LMP_COMPUTE_SUPERSATURATION_MONO_H
#define LMP_COMPUTE_SUPERSATURATION_MONO_H

#include "compute_cluster_temps.h"
#include "compute.h"
#include "region.h"

namespace LAMMPS_NS {

class ComputeSupersaturationMono : public Compute {
 public:
  ComputeSupersaturationMono(class LAMMPS *lmp, int narg, char **arg);
  ~ComputeSupersaturationMono() noexcept(true) override;
  void init() override;
  double compute_scalar() override;
  void compute_local() override;
  double memory_usage() override;

  double local_scalar;            // local supersaturation
  int local_monomers;             // number of local monomers
  bigint global_monomers{};       // number of global monomers
  double execute_func() const;    // monomer number density at saturation curve
  int *mono_idx{};                // ids of local monomers

 private:
  Region *region = nullptr;
  Compute *compute_neighs = nullptr;
  Compute *compute_temp = nullptr;
  ComputeClusterTemp *compute_cltemp = nullptr;

  bool use_t1;
  int nloc{};    // number of elements in mono_idx
  double coeffs[2]{};
};

}    // namespace LAMMPS_NS

#endif
#endif
