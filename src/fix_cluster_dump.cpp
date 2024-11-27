/*
 ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "fix_cluster_dump.h"
#include "fmt/base.h"
#include <cstring>

#include "comm.h"
#include "compute_cluster_size.h"
#include "error.h"
#include "fix.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define DUMP_FLOAT_PRECISION ",{:.5f}"

/* ---------------------------------------------------------------------- */

FixClusterDump::FixClusterDump(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), num_vectors(0), num_scalars(0)
{
  nevery = 1;

  if (narg < 8) { utils::missing_cmd_args(FLERR, "cluster/dump", error); }

  // Parse arguments //

  // Get the nevery
  nevery = utils::inumeric(FLERR, arg[3], true, lmp);
  if (nevery < 1) { error->all(FLERR, "nevery for {} nust be greater than 0", style); }

  ComputeClusterSize *compute_cluster_size =
      dynamic_cast<ComputeClusterSize *>(modify->get_compute_by_id(arg[4]));
  if (compute_cluster_size == nullptr) {
    error->all(FLERR, "{}: Cannot find compute cluster/size with id: {}", style, arg[4]);
  }
  size_cutoff = compute_cluster_size->get_size_cutoff();

  // Get the cutoff write size
  int const t_size_cutoff = utils::inumeric(FLERR, arg[5], true, lmp);
  if (t_size_cutoff < 1) { error->all(FLERR, "size_cutoff for {} must be greater than 0", style); }
  size_cutoff = MIN(size_cutoff, t_size_cutoff);

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
      fmt::print(file_scalars, "ntimestep");
      for (int i = 0; i < num_scalars; ++i) {
        const char *current_arg = arg[i + scalar_start_arg];
        compute_scalars[i] = modify->get_compute_by_id(current_arg);
        if (compute_scalars[i] == nullptr) {
          error->all(FLERR, "{}: cannot find Compute with id: {}", style, current_arg);
        }
        fmt::print(file_scalars, ",{}", current_arg);
      }
      fmt::print(file_scalars, "\n");
      ::fflush(file_scalars);
    }
  }
}

/* ---------------------------------------------------------------------- */

FixClusterDump::~FixClusterDump() noexcept(true)
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

/* ---------------------------------------------------------------------- */

void FixClusterDump::init()
{
  if ((modify->get_fix_by_style(style).size() > 1) && (comm->me == 0)) {
    error->warning(FLERR, "More than one fix {}", style);
  }
}

/* ---------------------------------------------------------------------- */

int FixClusterDump::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixClusterDump::end_of_step()
{
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

  if (comm->me == 0) {
    if (num_vectors > 0) {
      for (int i = 0; i < num_vectors; ++i) {
        FILE *current_file = file_vectors[i];
        double *current_vec = compute_vectors[i]->vector;
        fmt::print(current_file, "{}", update->ntimestep);
        for (int j = 1; j <= size_cutoff; ++j) { fmt::print(current_file, ",{}", current_vec[j]); }
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
}    // void FixClusterCrush::end_of_step()
