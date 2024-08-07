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

#include "compute.h"
#include "region.h"

namespace LAMMPS_NS {

class ComputeSupersaturationMono : public Compute {
 public:
  ComputeSupersaturationMono(class LAMMPS *, int, char **);
  ~ComputeSupersaturationMono() override;
  void init() override;
  double compute_scalar() override;
  void compute_local() override;
  double memory_usage() override;

  double local_scalar;
 private:
  double xlo, ylo, zlo, xhi, yhi, zhi;
  double lamda[3];
  double *boxlo, *boxhi;
  double sublo[3], subhi[3];    // epsilon-extended proc sub-box for adding atoms

  Region *region = nullptr;
  double coeffs[2];
  Compute *compute_neighs = nullptr;
  Compute *compute_temp = nullptr;
  int local_monomers;
  bigint global_monomers;

  double execute_func();
};

}    // namespace LAMMPS_NS

#endif
#endif
