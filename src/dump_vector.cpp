#include "dump_vector.h"
#include "compute.h"
#include "error.h"
#include "memory.h"
#include "update.h"

#include <fstream>
#include <sstream>

using namespace LAMMPS_NS;

DumpVector::DumpVector(LAMMPS *lmp, int narg, char **arg) : Dump(lmp, narg, arg) {
  if (narg < 4) error->all(FLERR, "Illegal dump vector command");
  vector_index = utils::inumeric(FLERR, arg[3], false, lmp);
  vector_data = nullptr;
}

DumpVector::~DumpVector() {
  delete[] vector_data;
}

void DumpVector::init_style() {
  // Initialize the vector data
  vector_data = new double[atom->nmax];
}

void DumpVector::write_header(bigint ndump) {
  // Write the header for the CSV file
  if (filewriter) {
    fprintf(fp, "TIMESTEP,%lld\n", update->ntimestep);
    fprintf(fp, "NUMBER OF ATOMS,%lld\n", ndump);
    fprintf(fp, "VECTOR DATA\n");
  }
}

void DumpVector::pack(tagint *ids) {
  // Pack the vector data into the buffer
  Compute *compute = modify->get_compute_by_index(vector_index);
  if (!compute) error->all(FLERR, "Compute not found for vector dump");
  
  compute->compute_peratom();
  double *data = compute->vector_atom;
  for (int i = 0; i < atom->nlocal; i++) {
    vector_data[i] = data[i];
  }
}

void DumpVector::write_data(int n, double *mybuf) {
  // Write the vector data to the CSV file
  if (filewriter) {
    for (int i = 0; i < n; i++) {
      fprintf(fp, "%g\n", vector_data[i]);
    }
  }
}
