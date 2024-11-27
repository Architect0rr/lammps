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
  Compute *my_compute; // pointer to the vector data
  int write_cutoff;

  void init_style() override;
  void write_header(bigint) override;
  void pack(tagint *) override;
  void write_data(int, double *) override;
  void write_csv(int, double *);
};

} // namespace LAMMPS_NS

#endif
#endif
