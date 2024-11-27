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
  extvector = 0;

  if (narg < 3) { utils::missing_cmd_args(FLERR, "compute cluster/temp", error); }

  // Parse arguments //

  // Get cluster/size compute
  compute_cluster_size = dynamic_cast<ComputeClusterSize *>(lmp->modify->get_compute_by_id(arg[3]));
  if (compute_cluster_size == nullptr) {
    error->all(FLERR, "{}: Cannot find compute with style 'cluster/size' with id: {}", style,
               arg[3]);
  }

  // Get the critical size
  size_cutoff = compute_cluster_size->get_size_cutoff();
  if ((narg >= 4) && (::strcmp(arg[4], "inherit") != 0)) {
    int t_size_cutoff = utils::inumeric(FLERR, arg[4], true, lmp);
    if (t_size_cutoff < 1) {
      error->all(FLERR, "size_cutoff for {} must be greater than 0", style);
    }
    size_cutoff = MIN(size_cutoff, t_size_cutoff);
  }

  // Get ke/atom compute
  auto computes = lmp->modify->get_compute_by_style("cluster/ke");
  if (computes.empty()) {
    error->all(FLERR, "{}: Cannot find compute with style 'cluster/ke'", style);
  }
  compute_cluster_ke = computes[0];

  size_vector = size_cutoff + 1;
  memory->create(temp, size_vector + 1, "cluster/temp:temp");
  vector = temp;
}

/* ---------------------------------------------------------------------- */

ComputeClusterTemp::~ComputeClusterTemp() noexcept(true)
{
  if (temp != nullptr) { memory->destroy(temp); }
}

/* ---------------------------------------------------------------------- */

void ComputeClusterTemp::init()
{
  if ((modify->get_compute_by_style(style).size() > 1) && (comm->me == 0)) {
    error->warning(FLERR, "More than one compute {}", style);
  }
}

/* ---------------------------------------------------------------------- */

void ComputeClusterTemp::compute_vector()
{
  invoked_vector = update->ntimestep;

  if (compute_cluster_size->invoked_vector != update->ntimestep) {
    compute_cluster_size->compute_vector();
  }

  if (compute_cluster_ke->invoked_vector != update->ntimestep) {
    compute_cluster_ke->compute_vector();
  }

  const double *const kes = compute_cluster_ke->vector;
  ::memset(temp, 0.0, size_vector * sizeof(double));
  const double *const dist = compute_cluster_size->vector;
  for (int i = 0; i < size_cutoff; ++i) {
    if (dist[i] > 0) { temp[i] /= i * domain->dimension / 2; }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeClusterTemp::memory_usage()
{
  return static_cast<double>(size_vector * sizeof(double));
}
