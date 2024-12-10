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

#include "compute_cluster_cf.h"
#include "compute_cluster_size_ext.h"
#include "compute_cf_atom.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstring>
#include <unordered_map>

using namespace LAMMPS_NS;
using NUCC::cspan;

/* ---------------------------------------------------------------------- */

ComputeClusterCF::ComputeClusterCF(LAMMPS* lmp, int narg, char** arg) : Compute(lmp, narg, arg)
{
  array_flag      = 1;
  size_array_cols = 0;
  size_array_rows = 0;
  extarray        = 0;
  local_flag      = 1;
  size_local_rows = 0;
  size_local_cols = 0;

  if (narg < 5) { utils::missing_cmd_args(FLERR, "compute cf/cluster/avg", error); }

  // Parse arguments //

  // Get cluster/size compute
  compute_cluster_size = dynamic_cast<ComputeClusterSizeExt*>(lmp->modify->get_compute_by_id(arg[3]));
  if (compute_cluster_size == nullptr) { error->all(FLERR, "compute {}: Cannot find compute with style 'cluster/size' with id: {}", style, arg[3]); }

  compute_rdf_atom = dynamic_cast<ComputeCFAtom*>(lmp->modify->get_compute_by_id(arg[4]));
  if (compute_cluster_size == nullptr) { error->all(FLERR, "compute {}: Cannot find compute with style 'cf/atom' with id: {}", style, arg[4]); }

  size_cutoff = compute_cluster_size->get_size_cutoff();

  int iarg    = 5;
  while (iarg < narg) {
    if (iarg + 2 > narg) { error->all(FLERR, "Illegal command", style); }
    if (::strcmp(arg[iarg], "cut") == 0) {
      int t_size_cutoff = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (t_size_cutoff < 1) { error->all(FLERR, "size_cutoff for compute {} must be greater than 0", style); }
      if (t_size_cutoff > size_cutoff) {
        error->all(FLERR,
                   "size_cutoff for compute {} cannot be greater than it of "
                   "compute cluster/size",
                   style);
      }
      iarg += 2;
    } else {
      error->all(FLERR, "Illegal {} command", style);
    }
  }
}

/* ---------------------------------------------------------------------- */

ComputeClusterCF::~ComputeClusterCF() noexcept(true)
{
  memory->destroy(cf);
  memory->destroy(cf_local);
}

/* ---------------------------------------------------------------------- */

void ComputeClusterCF::init()
{
  if ((modify->get_compute_by_style(style).size() > 1) && (comm->me == 0)) { error->warning(FLERR, "More than one compute {}", style); }

  size_local_rows = size_cutoff + 1;
  size_local_cols = compute_rdf_atom->size_peratom_cols;
  array_local     = memory->create(cf_local, size_local_rows, size_local_cols, "cf/cluster/avg:cf_local");

  size_array_rows = size_cutoff + 1;
  size_array_cols = compute_rdf_atom->size_peratom_cols;
  array           = memory->create(cf, size_array_rows, size_array_cols, "cf/cluster/avg:cf");
}

/* ---------------------------------------------------------------------- */

void ComputeClusterCF::compute_vector()
{
  invoked_vector = update->ntimestep;

  compute_local();

  for (int i = 0; i < size_array_rows; ++i) {
    ::memset(cf[i], 0.0, size_array_cols * sizeof(double));
    ::MPI_Allreduce(cf_local, cf, size_array_cols, MPI_DOUBLE, MPI_SUM, world);
  }

  const double* dist = compute_cluster_size->vector;
  for (int i = 0; i < size_array_rows; ++i) {
    if (dist[i] > 0) {
      for (int j = 0; j < size_array_cols; ++j) { cf[i][j] /= dist[i]; }
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeClusterCF::compute_local()
{
  invoked_local = update->ntimestep;

  if (compute_cluster_size->invoked_vector != update->ntimestep) { compute_cluster_size->compute_vector(); }

  if (compute_rdf_atom->invoked_peratom != update->ntimestep) { compute_rdf_atom->compute_peratom(); }

  for (int i = 0; i < size_local_rows; ++i) { ::memset(cf_local[i], 0.0, size_local_cols * sizeof(double)); }

  // const double** const peratomcf = compute_rdf_atom->array_atom;
  // for (const auto& [size, vec] : compute_cluster_size->cIDs_by_size) {
  //   if (size < size_cutoff) {
  //     for (const tagint cid : vec) {
  //       for (const tagint pid : compute_cluster_size->atoms_by_cID[cid]) {
  //         for (int i = 0; i < size_local_cols; ++i) { cf_local[size][i] += peratomcf[pid][i]; }
  //       }
  //     }
  //   }
  // }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeClusterCF::memory_usage()
{
  std::size_t sum = size_array_rows * (size_array_cols * sizeof(double) + sizeof(double*));
  sum += size_local_rows * (size_local_cols * sizeof(double) + sizeof(double*));
  return static_cast<double>(sum);
}
