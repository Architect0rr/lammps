#ifdef DUMP_CLASS
// clang-format off
DumpStyle(vector,DumpVector);
// clang-format on
#else

#ifndef LMP_DUMP_VECTOR_H
#define LMP_DUMP_VECTOR_H

#include "dump.h"
#include "compute.h"

namespace LAMMPS_NS {

class DumpVector : public Dump {
 public:
  DumpVector(LAMMPS *, int, char **);
  ~DumpVector() override;

 protected:
  Compute **computes; // array to store pointers to the vector data computes
  int num_computes;   // number of computes
  double *vector_data; // pointer to store vector data
  int write_cutoff;    // number of elements to write

  void init_style() override;
  void write_header(bigint) override;
  void pack(tagint *) override;
  void write_data(int, double *) override;
};

} // namespace LAMMPS_NS

#endif
#endif
