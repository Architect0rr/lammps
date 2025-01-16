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

// TODO: NUCC FILE

#include "compute_cluster_enthropy.h"
#include "compute_cluster_size_ext.h"

#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstring>
#include <unordered_map>

using namespace LAMMPS_NS;

constexpr double planck_constant = 0.18292026;

/* ---------------------------------------------------------------------- */

ComputeClusterEnthropy::ComputeClusterEnthropy(LAMMPS* lmp, int narg, char** arg) : Compute(lmp, narg, arg)
{

  // vector_flag = 1;
  array_flag      = 1;
  // size_vector = 0;
  // size_vector_variable = 1;
  // extvector = 0;
  extarray        = 0;
  local_flag      = 1;
  size_local_rows = 0;
  size_local_cols = 4;

  if (narg < 4) { utils::missing_cmd_args(FLERR, "compute cluster/temp", error); }

  // Parse arguments //

  // Get cluster/size compute
  compute_cluster_size = dynamic_cast<ComputeClusterSizeExt*>(lmp->modify->get_compute_by_id(arg[3]));
  if (compute_cluster_size == nullptr) {
    error->all(FLERR, "{}: Cannot find compute with style 'cluster/size' with id: {}", style, arg[3]);
  }

  // Get entropy/atom compute
  compute_entropy_atom = lmp->modify->get_compute_by_id(arg[4]);
  if (compute_entropy_atom == nullptr) { error->all(FLERR, "{}: Cannot find compute with id '{}'", style, arg[4]); }

  size_cutoff = compute_cluster_size->get_size_cutoff();
  if ((narg >= 6) && (::strcmp(arg[5], "inherit") != 0)) {
    int t_size_cutoff = utils::inumeric(FLERR, arg[5], true, lmp);
    if (t_size_cutoff < 1) { error->all(FLERR, "{}: size_cutoff must be greater than 0", style); }
    if (t_size_cutoff > size_cutoff) { error->all(FLERR, "{}: size_cutoff cannot be greater than it of compute sizecluster", style); }
  }

  size_local_rows = size_cutoff + 1;
  local_enth.create(memory, size_local_rows + 1, "compute:cluster/enthropy:temp");
  vector_local = local_enth.data();

  size_vector  = size_cutoff + 1;
  enth.create(memory, size_vector + 1, "compute:cluster/enthropy:temp");
  vector = enth.data();
}

/* ---------------------------------------------------------------------- */

ComputeClusterEnthropy::~ComputeClusterEnthropy() noexcept(true)
{
  local_enth.destroy(memory);
  enth.destroy(memory);
}

/* ---------------------------------------------------------------------- */

void ComputeClusterEnthropy::init()
{
  if ((modify->get_compute_by_style(style).size() > 1) && (comm->me == 0)) { error->warning(FLERR, "More than one compute {}", style); }
}

/* ---------------------------------------------------------------------- */

void ComputeClusterEnthropy::compute_vector()
{
  invoked_vector = update->ntimestep;

  compute_local();

  ::MPI_Allreduce(local_enth.data(), enth.data(), size_vector, MPI_DOUBLE, MPI_SUM, world);

  const double* dist = compute_cluster_size->vector;
  for (int i = 0; i < size_vector; ++i) { enth[i] /= dist[i]; }
}

/* ---------------------------------------------------------------------- */

void ComputeClusterEnthropy::compute_local()
{
  invoked_local = update->ntimestep;

  if (compute_cluster_size->invoked_vector != update->ntimestep) { compute_cluster_size->compute_vector(); }

  const double* enths = compute_entropy_atom->vector_atom;
  local_enth.reset();

  int nclusters = dynamic_cast<ComputeClusterSizeExt*>(compute_cluster_size)->get_cluster_map().size();
  const auto& clusters = dynamic_cast<ComputeClusterSizeExt*>(compute_cluster_size)->get_clusters();
  for (int i = 0; i < nclusters; ++i) {
    const auto& clstr = clusters[i];
    const auto& atoms = clstr.atoms();
    for (int j = 0; j < clstr.l_size; ++j) {
      auto a = local_enth[clstr.g_size];
      local_enth[clstr.g_size] += enths[atoms[j]];
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeClusterEnthropy::memory_usage()
{
  return static_cast<double>((size_local_rows + size_vector) * sizeof(double));
}
