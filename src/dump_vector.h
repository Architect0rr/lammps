#ifndef LMP_DUMP_VECTOR_H
#define LMP_DUMP_VECTOR_H

#include "dump.h"

namespace LAMMPS_NS {

class DumpVector : public Dump {
 public:
  DumpVector(LAMMPS *, int, char **);
  ~DumpVector() override;

 protected:
  int vector_index;  // index of the vector to dump
  double *vector_data; // pointer to the vector data

  void init_style() override;
  void write_header(bigint) override;
  void pack(tagint *) override;
  void write_data(int, double *) override;
  void write_csv(int, double *);
};

} // namespace LAMMPS_NS

#endif
