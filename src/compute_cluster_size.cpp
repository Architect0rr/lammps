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
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeClusterSize::ComputeClusterSize(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), nloc(0), nloc_atom(0), peratom_size(nullptr), dist(nullptr),
    nc_global(0)
{
  vector_flag = 1;
  extvector = 0;
  size_vector = 0;
  size_vector_variable = 1;

  peratom_flag = 1;
  size_peratom_cols = 0;

  if (narg < 4) { utils::missing_cmd_args(FLERR, "compute cluster/size", error); }

  // Parse arguments //

  // Get cluster/atom compute
  compute_cluster_atom = lmp->modify->get_compute_by_id(arg[3]);
  if (compute_cluster_atom == nullptr) {
    error->all(
        FLERR,
        "compute cluster/size: Cannot find compute with style 'cluster/atom' with given id: {}",
        arg[3]);
  }

  // Get the critical size
  size_cutoff = utils::inumeric(FLERR, arg[4], true, lmp);
  if (size_cutoff < 1) {
    error->all(FLERR, "size_cutoff for compute cluster/size must be greater than 0");
  }

  size_vector = size_cutoff + 1;
  dist = memory->create(dist, (size_vector + 1) * sizeof(double), "compute:cluster/size:dist");
  vector = dist;
}

/* ---------------------------------------------------------------------- */

ComputeClusterSize::~ComputeClusterSize() noexcept(true)
{
  if (dist != nullptr) { memory->destroy(dist); }
  if (peratom_size != nullptr) { memory->destroy(peratom_size); }
}

/* ---------------------------------------------------------------------- */

void ComputeClusterSize::init()
{
  if ((modify->get_compute_by_style(style).size() > 1) && (comm->me == 0)) {
    error->warning(FLERR, "More than one compute {}", style);
  }

  nloc_atom = atom->nlocal;
  peratom_size = memory->create(peratom_size, nloc_atom * sizeof(double), "cluster_size:peratom");
  ::memset(peratom_size, 0.0, nloc_atom * sizeof(double));
  vector_atom = peratom_size;

  initialized_flag = 1;
}

/* ---------------------------------------------------------------------- */

void ComputeClusterSize::compute_vector()
{
  invoked_vector = update->ntimestep;

  if (compute_cluster_atom->invoked_peratom != update->ntimestep) {
    compute_cluster_atom->compute_peratom();
  }

  const double *cluster_ids = compute_cluster_atom->vector_atom;

  if (atom->nlocal > nloc) {
    nloc = atom->nlocal;
    atoms_by_cID.reserve(nloc);
    cIDs_by_size.reserve(nloc);
  }

  // Clear buffers
  atoms_by_cID.clear();
  cIDs_by_size.clear();

  // Sort atom IDs by cluster IDs
  for (int i = 0; i < atom->nlocal; ++i) {
    if ((atom->mask[i] & groupbit) != 0) {
      atoms_by_cID[static_cast<bigint>(cluster_ids[i])].emplace_back(i);
    }
  }

  nc_global = 0;
  ::memset(dist, 0.0, size_vector * sizeof(double));

  // Sum cluster size over all procs
  bigint l_size = 0;    // local size of cluster
  bigint g_size = 0;    // global size of cluster
  for (bigint i = 1; i <= atom->natoms; ++i) {
    l_size = atoms_by_cID.count(i) == 0 ? 0 : atoms_by_cID[i].size();
    ::MPI_Allreduce(&l_size, &g_size, 1, MPI_LMP_BIGINT, MPI_SUM, world);
    if (l_size > 0) { cIDs_by_size[g_size].emplace_back(i); }
    if (g_size > 0) {
      if (g_size < size_cutoff) { dist[g_size] += 1; }
      ++nc_global;
    }
  }

  compute_peratom();
}

/* ---------------------------------------------------------------------- */

void ComputeClusterSize::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  if ((nloc_atom < atom->nlocal) && (peratom_size != nullptr)) { memory->destroy(peratom_size); }

  if ((nloc_atom < atom->nlocal) && (peratom_size == nullptr)) {
    nloc_atom = atom->nlocal;
    peratom_size = memory->create(peratom_size, nloc_atom * sizeof(double), "cluster_size:peratom");
    ::memset(peratom_size, 0.0, nloc_atom * sizeof(double));
    vector_atom = peratom_size;
  }

  for (auto [size, cIDs] : cIDs_by_size) {
    for (auto cID : cIDs) {
      for (auto pID : atoms_by_cID[cID]) { peratom_size[pID] = size; }
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of maps and dist
------------------------------------------------------------------------- */

double ComputeClusterSize::memory_usage()
{
  size_t const atoms_by_cID_elementsMemory = atoms_by_cID.size() *
      (sizeof(std::string) + sizeof(int) + 2 * sizeof(void *));    // Assuming 2 pointers per node;
  size_t const atoms_by_cID_bucketOverhead =
      atoms_by_cID.bucket_count() * sizeof(void *);    // Assuming each bucket is just a pointer
  size_t const cIDs_by_size_elementsMemory = cIDs_by_size.size() *
      (sizeof(std::string) + sizeof(int) + 2 * sizeof(void *));    // Assuming 2 pointers per node;
  size_t const cIDs_by_size_bucketOverhead =
      cIDs_by_size.bucket_count() * sizeof(void *);    // Assuming each bucket is just a pointer
  return size_vector * sizeof(double) +
      static_cast<double>(atoms_by_cID_elementsMemory + atoms_by_cID_bucketOverhead +
                          cIDs_by_size_elementsMemory + cIDs_by_size_bucketOverhead);
}

/* ---------------------------------------------------------------------- */

int ComputeClusterSize::get_size_cutoff() const noexcept(true)
{
  return size_cutoff;
}
