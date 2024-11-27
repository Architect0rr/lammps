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

#ifdef DUMP_CLASS
// clang-format off
DumpStyle(vector,DumpVector);
// clang-format on
#else

#ifndef LMP_DUMP_VECTOR_H
#define LMP_DUMP_VECTOR_H

#include "compute.h"
#include "dump.h"
#include "memory.h"

namespace LAMMPS_NS {

class DumpVector : public Dump {
 public:
  DumpVector(LAMMPS *, int, char **);
  ~DumpVector() override;

 protected:
  Compute **compute_vectors{};    // array to store pointers to the vector data computes
  Compute **compute_scalars{};

  int num_vectors{};        // number of computes
  double *vector_data{};    // pointer to store vector data
  FILE **file_vectors{};
  int write_cutoff;    // number of elements to write
  int num_scalars{};

  FILE *file_scalars;

  void init_style() override;
  void write_header(bigint) override;
  void pack(tagint *) override {}
  void write() override;
  void write_data(int, double *) override {}
  template <typename TYPE> inline TYPE **create_ptr_array(TYPE **&array, int n, const char *name)
  {
    array = n <= 0 ? nullptr : static_cast<TYPE **>(memory->smalloc(sizeof(TYPE *) * n, name));
    return array;
  }
};

}    // namespace LAMMPS_NS

#endif
#endif
