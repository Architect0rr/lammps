#ifdef DUMP_CLASS
// clang-format off
DumpStyle(vector,DumpVector);
// clang-format on
#else

#ifndef LMP_DUMP_VECTOR_H
#define LMP_DUMP_VECTOR_H

#include "compute.h"
#include "dump.h"

namespace LAMMPS_NS {

class DumpVector : public Dump {
 public:
  DumpVector(LAMMPS *, int, char **);
  ~DumpVector() override;

 protected:
  Compute **computes;     // array to store pointers to the vector data computes
  int num_computes;       // number of computes
  double *vector_data;    // pointer to store vector data
  int write_cutoff;       // number of elements to write

  void init_style() override;
  void write_header(bigint) override;
  void pack(tagint *) override;
  void write_data(int, double *) override;
  void openfile(int compute_index);    // New method to open file based on compute index
  template <typename TYPE> inline TYPE **create_ptr_array(TYPE **&array, int n, const char *name)
  {
    array = n <= 0 ? nullptr : static_cast<TYPE **>(memory->smalloc(sizeof(TYPE *) * n, name));
    return array;
  }
};

}    // namespace LAMMPS_NS

#endif
#endif
