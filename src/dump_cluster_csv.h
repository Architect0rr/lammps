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
DumpStyle(cluster/csv,DumpClusterCSV);
// clang-format on
#else

#ifndef LMP_DUMP_CLUSTER_CSV_H
#define LMP_DUMP_CLUSTER_CSV_H

#include "dump.h"
#include "fmt/base.h"
#include <cstring>

#include "comm.h"
#include "compute_cluster_size.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

namespace LAMMPS_NS {

class DumpClusterCSV : public Dump {
 public:
  DumpClusterCSV(LAMMPS *lmp, int narg, char **arg);
  ~DumpClusterCSV() noexcept(true) override;
  void init_style() override;
  void write() override;

 protected:
  Compute **compute_vectors{};
  Compute **compute_scalars{};
  int num_vectors;
  int num_scalars;

  FILE **file_vectors{};
  FILE *file_scalars;

  int size_cutoff;

  template <typename TYPE> inline TYPE **create_ptr_array(TYPE **&array, int n, const char *name)
  {
      array = n <= 0 ? nullptr : static_cast<TYPE **>(memory->smalloc(sizeof(TYPE *) * n, name));
      return array;
  }
};

}    // namespace LAMMPS_NS

#endif
#endif
