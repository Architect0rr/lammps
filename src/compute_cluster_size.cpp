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

#include "compute_cluster_size.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <string.h>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeClusterSize::ComputeClusterSize(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), nloc(0), nc_global(0), dist(nullptr)
{
  vector_flag = 1;
  extvector = 0;
  size_vector = 0;
  size_vector_variable = 1;

  if (narg < 5) utils::missing_cmd_args(FLERR, "compute cluster/size", error);

  // Parse arguments //

  // Target region
  region = domain->get_region_by_id(arg[3]);
  if (region == nullptr) {
    error->all(FLERR, "compute cluster/size: Cannot find target region {}", arg[3]);
  }

  // Get cluster/atom compute
  compute_cluster_atom = lmp->modify->get_compute_by_id(arg[4]);
  if (compute_cluster_atom == nullptr) {
    error->all(
        FLERR,
        "compute cluster/size: Cannot find compute with style 'cluster/atom' with given id: {}",
        arg[4]);
  }
}

/* ---------------------------------------------------------------------- */

ComputeClusterSize::~ComputeClusterSize()
{
  if (dist != nullptr) { memory->destroy(dist); }
}

/* ---------------------------------------------------------------------- */

void ComputeClusterSize::init()
{
  if (modify->get_compute_by_style(style).size() > 1)
    if (comm->me == 0) error->warning(FLERR, "More than one compute {}", style);
}

/* ---------------------------------------------------------------------- */

void ComputeClusterSize::compute_vector()
{
  invoked_vector = update->ntimestep;

  if (compute_cluster_atom->invoked_peratom != update->ntimestep) {
    compute_cluster_atom->compute_peratom();
  }

  double *cluster_ids = compute_cluster_atom->vector_atom;

  if (size_vector != atom->natoms + 1 && dist != nullptr) { memory->destroy(dist); }
  if (size_vector != atom->natoms + 1 || dist == nullptr) {
    size_vector = atom->natoms + 1;
    memory->create(dist, size_vector, "compute:cluster/size:dist");
    vector = dist;
  }

  if (atom->nlocal > nloc) {
    nloc = atom->nlocal;
    atoms_by_cID.reserve(nloc);
    cIDs_by_size.reserve(nloc);
    // unique_cIDs.reserve(nloc);
  }

  // Clear buffers
  atoms_by_cID.clear();
  cIDs_by_size.clear();
  // unique_cIDs.clear();

  // Sort atom IDs by cluster IDs
  for (int i = 0; i < atom->nlocal; ++i) {
    if (atom->mask[i] & groupbit) {
      tagint cid = static_cast<tagint>(cluster_ids[i]);
      // unique_cIDs.emplace(cid);
      atoms_by_cID[cid].emplace_back(i);
    }
  }

  nc_global = 0;
  memset(dist, 0.0, size_vector * sizeof(double));
  // Sum cluster size over all procs
  bigint l_size = 0;    // local size of cluster
  bigint g_size = 0;    // global size of cluster
  for (bigint i = 1; i <= atom->natoms; ++i) {
    l_size = 0 ? atoms_by_cID.count(i) == 0 : atoms_by_cID[i].size();
    MPI_Allreduce(&l_size, &g_size, 1, MPI_INT, MPI_SUM, world);
    if (l_size > 0) { cIDs_by_size[g_size].emplace_back(i); }
    if (g_size > 0) {
      dist[g_size] += 1;
      ++nc_global;
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of maps and dist
------------------------------------------------------------------------- */

double ComputeClusterSize::memory_usage()
{
  return size_vector * sizeof(double);
}