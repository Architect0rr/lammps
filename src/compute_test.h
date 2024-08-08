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
ComputeStyle(test,ComputeTest);
// clang-format on
#else

#ifndef LMP_COMPUTE_TEST_H
#define LMP_COMPUTE_TEST_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeTest : public Compute {
 public:
  ComputeTest(class LAMMPS *, int, char **);
  ~ComputeTest() override;
  void init() override;
  double compute_scalar() override;
  void compute_vector() override;
  void compute_array() override;
  void compute_peratom() override;
  void compute_local() override;
  double memory_usage() override;

 private:
  int nloc = 0;
};

}    // namespace LAMMPS_NS

#endif
#endif