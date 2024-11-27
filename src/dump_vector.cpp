#include "dump_vector.h"
#include "compute.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

using namespace LAMMPS_NS;

DumpVector::DumpVector(LAMMPS *lmp, int narg, char **arg) : Dump(lmp, narg, arg)
{
  if (narg < 4) { error->all(FLERR, "Illegal dump vector command"); }

  num_computes = narg - 3;                                 // Number of computes passed in arguments
  create_ptr_array(computes, num_computes, "computes");    // Allocate memory for computes array

  for (int i = 0; i < num_computes; i++) {
    computes[i] = modify->get_compute_by_id(arg[i + 3]);
    if (computes[i] == nullptr) {
      error->all(FLERR, "{}: Cannot find compute with id: {}", style, arg[i + 3]);
    }
  }

  write_cutoff = 10;    // Set a default value for how many elements to write
  memory->create(vector_data, write_cutoff, "vector");    // Allocate memory for vector data
}

DumpVector::~DumpVector()
{
  if (vector_data != nullptr) { memory->destroy(vector_data); }
  if (computes != nullptr) { memory->destroy(computes); }
}

void DumpVector::init_style()
{
  // Initialize the vector data if needed
}

void DumpVector::write_header(bigint /*ndump*/)
{
  // Write the header for the CSV file for each compute
  if (filewriter != 0) {
    for (int i = 0; i < num_computes; i++) {
      openfile(i);    // Open a file for each compute
      fprintf(fp, "TIMESTEP,%ld\n", update->ntimestep);
      fprintf(fp, "VECTOR DATA\n");
    }
  }
}

void DumpVector::openfile(int compute_index)
{
  // Open a separate file for each compute based on its index
  std::string filename = fmt::format("dump_vector_{}.csv", compute_index);
  fp = fopen(filename.c_str(), "w");
  if (fp == nullptr) { error->one(FLERR, "Cannot open dump file: {}", filename); }
}

void DumpVector::pack(tagint * /*ids*/)
{
  for (int i = 0; i < num_computes; i++) {
    if (computes[i]->invoked_vector != update->ntimestep) { computes[i]->compute_vector(); }
    // Copy the vector data to the member variable
    for (int j = 0; j < write_cutoff; j++) { vector_data[j] = computes[i]->vector[j]; }
  }
}

void DumpVector::write_data(int /*n*/, double * /*mybuf*/)
{
  // Write the vector data to the CSV file for each compute
  if (filewriter != 0) {
    for (int i = 0; i < num_computes; i++) {
      for (int j = 0; j < write_cutoff; j++) { fprintf(fp, "%g\n", vector_data[j]); }
      fclose(fp);    // Close the file after writing
    }
  }
}
