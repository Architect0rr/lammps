/*
 ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "dump_vector.h"
#include "comm.h"
#include "compute.h"
#include "compute_cluster_size.h"
#include "error.h"
#include "fmt/base.h"
#include "memory.h"
#include "modify.h"
#include "update.h"
#include <cstring>

using namespace LAMMPS_NS;

#define DUMP_FLOAT_PRECISION ",{:.5f}"

/* ---------------------------------------------------------------------- */

DumpVector::DumpVector(LAMMPS *lmp, int narg, char **arg) : Dump(lmp, narg, arg)
{
  if (narg < 4) { error->all(FLERR, "Illegal dump vector command"); }
  clearstep = 1;

  ComputeClusterSize *compute_cluster_size =
      dynamic_cast<ComputeClusterSize *>(modify->get_compute_by_id(arg[4]));
  if (compute_cluster_size == nullptr) {
    error->all(FLERR, "{}: Cannot find compute cluster/size with id: {}", style, arg[4]);
  }
  write_cutoff = compute_cluster_size->get_size_cutoff();

  // Get the cutoff write size
  int const t_size_cutoff = utils::inumeric(FLERR, arg[5], true, lmp);
  if (t_size_cutoff < 1) { error->all(FLERR, "size_cutoff for {} must be greater than 0", style); }
  write_cutoff = MIN(write_cutoff, t_size_cutoff);

  constexpr int compute_list_start_arg = 6;
  bool count_vector = true;
  int scalar_start_arg = 0;
  for (int i = compute_list_start_arg; i < narg; ++i) {
    if (::strcmp(arg[i], "vector") == 0) {
      if (!count_vector) { error->all(FLERR, "{}: vectors if exists must be first", style); }
      continue;
    }
    if (::strcmp(arg[i], "scalar") == 0) {
      count_vector = false;
      scalar_start_arg = i + 1;
      continue;
    }
    if (count_vector) {
      ++num_vectors;
    } else {
      ++num_scalars;
    }
  }

  create_ptr_array(file_vectors, num_vectors, "vector_files");
  create_ptr_array(compute_vectors, num_vectors, "vector_computes");
  create_ptr_array(compute_scalars, num_scalars, "scalar_computes");

  if (comm->me == 0) {
    if (num_vectors > 0) {
      constexpr int vector_start_arg = 7;
      for (int i = 0; i < num_vectors; ++i) {
        const char *current_arg = arg[i + vector_start_arg];
        compute_vectors[i] = modify->get_compute_by_id(current_arg);
        if (compute_vectors[i] == nullptr) {
          error->all(FLERR, "{}: cannot find Compute with id: {}", style, current_arg);
        }
        file_vectors[i] = ::fopen(fmt::format("{}.csv", current_arg).c_str(), "a");
        if (file_vectors[i] == nullptr) {
          error->one(FLERR, "{}: Cannot open file {}: {}", style,
                     fmt::format("{}.csv", current_arg), utils::getsyserror());
        }
      }
    }

    if (num_scalars > 0) {
      file_scalars = ::fopen("scalars.csv", "a");
      if (file_scalars == nullptr) {
        error->one(FLERR, "{}: Cannot open file {}: {}", style, "scalars.csv",
                   utils::getsyserror());
      }

      for (int i = 0; i < num_scalars; ++i) {
        const char *current_arg = arg[i + scalar_start_arg];
        compute_scalars[i] = modify->get_compute_by_id(current_arg);
        if (compute_scalars[i] == nullptr) {
          error->all(FLERR, "{}: Cannot find compute with id: {}", style, current_arg);
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

DumpVector::~DumpVector()
{
  for (int i = 0; i < num_vectors; ++i) {
    if (file_vectors[i] != nullptr) {
      ::fflush(file_vectors[i]);
      ::fclose(file_vectors[i]);
    }
  }
  if (file_vectors != nullptr) { memory->destroy(file_vectors); }
  if (file_scalars != nullptr) {
    ::fflush(file_scalars);
    ::fclose(file_scalars);
  }

  if (compute_vectors != nullptr) { memory->destroy(compute_vectors); }
  if (compute_scalars != nullptr) { memory->destroy(compute_scalars); }
}

void DumpVector::init_style()
{
  // Initialize the vector data if needed
}

void DumpVector::write_header(bigint /*ndump*/)
{
  // Write the header for the CSV file for each compute
  if (filewriter != 0) {
    if (num_scalars > 0) {
      fmt::print(file_scalars, "ntimestep");
      for (int i = 0; i < num_scalars; ++i) {
        fmt::print(file_scalars, ",{}", compute_scalars[i]->id);
      }
      fmt::print(file_scalars, "\n");
      ::fflush(file_scalars);
    }
  }
}

/* ---------------------------------------------------------------------- */

void DumpVector::write()
{
  // Write the vector data to the CSV file for each compute
  for (int i = 0; i < num_vectors; ++i) {
    if (compute_vectors[i]->invoked_vector != update->ntimestep) {
      compute_vectors[i]->compute_vector();
    }
  }

  for (int i = 0; i < num_scalars; ++i) {
    if (compute_scalars[i]->invoked_scalar != update->ntimestep) {
      compute_scalars[i]->compute_scalar();
    }
  }

  if (filewriter != 0) {
    if (num_vectors > 0) {
      for (int i = 0; i < num_vectors; ++i) {
        FILE *current_file = file_vectors[i];
        double *current_vec = compute_vectors[i]->vector;
        fmt::print(current_file, "{}", update->ntimestep);
        for (int j = 1; j <= write_cutoff; ++j) { fmt::print(current_file, ",{}", current_vec[j]); }
        fmt::print(current_file, "\n");
        ::fflush(current_file);
      }
    }

    if (num_scalars > 0) {
      fmt::print(file_scalars, "{}", update->ntimestep);
      for (int i = 0; i < num_scalars; ++i) {
        fmt::print(file_scalars, DUMP_FLOAT_PRECISION, compute_scalars[i]->scalar);
      }
      fmt::print(file_scalars, "\n");
      ::fflush(file_scalars);
    }
  }
}
