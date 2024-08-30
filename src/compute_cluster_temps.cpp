/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_cluster_temps.h"
#include "compute_cluster_size.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstring>
#include <unordered_map>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeClusterTemp::ComputeClusterTemp(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg)
{

  vector_flag = 1;
  size_vector = 0;
  size_vector_variable = 1;
  extvector = 0;
  // local_flag = 1;
  size_local_rows = 0;
  size_local_cols = 0;

  if (narg < 4) { utils::missing_cmd_args(FLERR, "compute cluster/temp", error); }

  // Parse arguments //

  // Get cluster/size compute
  compute_cluster_size = dynamic_cast<ComputeClusterSize *>(lmp->modify->get_compute_by_id(arg[3]));
  if (compute_cluster_size == nullptr) {
    error->all(FLERR,
               "compute cluster/temp: Cannot find compute with style 'cluster/size' with id: {}",
               arg[3]);
  }

  // Get ke/atom compute
  auto computes = lmp->modify->get_compute_by_style("ke/atom");
  if (computes.empty()) {
    error->all(FLERR, "compute cluster/temp: Cannot find compute with style 'ke/atom'");
  }
  compute_ke_atom = computes[0];
}

/* ---------------------------------------------------------------------- */

ComputeClusterTemp::~ComputeClusterTemp()
{
  if (local_temp != nullptr) { memory->destroy(local_temp); }
  if (temp != nullptr) { memory->destroy(temp); }
}

/* ---------------------------------------------------------------------- */

void ComputeClusterTemp::init()
{
  if (modify->get_compute_by_style(style).size() > 1) {
    if (comm->me == 0) { error->warning(FLERR, "More than one compute {}", style); }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeClusterTemp::compute_vector()
{
  invoked_vector = update->ntimestep;

  if (size_vector != atom->natoms + 1 && temp != nullptr) { memory->destroy(temp); }
  if (size_vector != atom->natoms + 1 || temp == nullptr) {
    size_vector = atom->natoms + 1;
    memory->create(temp, size_vector, "compute:cluster/temp:temp");
    vector = temp;
  }

  compute_local();

  MPI_Allreduce(local_temp, temp, size_vector, MPI_DOUBLE, MPI_SUM, world);

  double *dist = compute_cluster_size->vector;
  for (tagint i = 0; i < size_vector; ++i) { temp[i] /= (dist[i] * i - 1) * domain->dimension; }
}

/* ---------------------------------------------------------------------- */

void ComputeClusterTemp::compute_local()
{
  invoked_local = update->ntimestep;

  if (compute_cluster_size->invoked_vector != update->ntimestep) {
    compute_cluster_size->compute_vector();
  }

  if (compute_ke_atom->invoked_peratom != update->ntimestep) { compute_ke_atom->compute_peratom(); }

  if (size_local_rows != atom->natoms + 1 && local_temp != nullptr) { memory->destroy(local_temp); }
  if (size_local_rows != atom->natoms + 1 || local_temp == nullptr) {
    size_local_rows = atom->natoms + 1;
    memory->create(local_temp, size_local_rows, "compute:cluster/temp:temp");
    vector_local = local_temp;
  }

  double *kes = compute_ke_atom->vector_atom;
  memset(local_temp, 0.0, size_local_rows * sizeof(double));

  for (const auto &[size, vec] : compute_cluster_size->cIDs_by_size) {
    for (const tagint cid : vec) {
      for (const tagint pid : compute_cluster_size->atoms_by_cID[cid]) {
        local_temp[size] += 2 * kes[pid];
      }
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeClusterTemp::memory_usage()
{
  return (size_local_rows + size_vector) * sizeof(double);
}