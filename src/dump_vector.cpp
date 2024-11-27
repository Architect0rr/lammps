#include "dump_vector.h"
#include "compute.h"
#include "error.h"
#include "modify.h"
#include "memory.h"
#include "update.h"

#include <fstream>
#include <sstream>

using namespace LAMMPS_NS;

DumpVector::DumpVector(LAMMPS *lmp, int narg, char **arg) : Dump(lmp, narg, arg) {

  if (narg < 4) error->all(FLERR, "Illegal dump vector command");

  my_compute = modify->get_compute_by_id(arg[4]);
  if (my_compute == nullptr) {
    error->all(FLERR, "{}: Cannot find compute with id: {}", style, arg[4]);
  }

  write_cutoff = 10;
}

DumpVector::~DumpVector() {}

void DumpVector::init_style() {
  // Initialize the vector data
}

void DumpVector::write_header(bigint ndump) {
  // Write the header for the CSV file
  if (filewriter) {
    fprintf(fp, "TIMESTEP,%lld\n", update->ntimestep);
    fprintf(fp, "VECTOR DATA\n");
  }
}

void DumpVector::pack(tagint *ids) {
  if (my_compute->invoked_vector != update->ntimestep) {
    my_compute->compute_vector();
  }
}

void DumpVector::write_data(int n, double *mybuf) {
  // Write the vector data to the CSV file
  if (filewriter) {
    for (int i = 0; i < write_cutoff; i++) {
      fprintf(fp, "%g\n", my_compute->vector[i]);
    }
  }
}
