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

#include "compute_cluster_pe.h"
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

ComputeClusterPE::ComputeClusterPE(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg)
{
  vector_flag = 1;
  size_vector = 0;
  extvector = 0;
  local_flag = 1;
  size_local_rows = 0;
  size_local_cols = 0;

  if (narg < 3) { utils::missing_cmd_args(FLERR, "compute cluster/pe", error); }

  // Parse arguments //

  // Get cluster/size compute
  compute_cluster_size = dynamic_cast<ComputeClusterSize *>(lmp->modify->get_compute_by_id(arg[3]));
  if (compute_cluster_size == nullptr) {
    error->all(FLERR, "compute {}: Cannot find compute with style 'cluster/size' with id: {}",
               style, arg[3]);
  }

  // Get the critical size
  size_cutoff = compute_cluster_size->get_size_cutoff();
  if ((narg >= 4) && (::strcmp(arg[4], "inherit") != 0)) {
    int t_size_cutoff = utils::inumeric(FLERR, arg[4], true, lmp);
    if (t_size_cutoff < 1) {
      error->all(FLERR, "size_cutoff for compute {} must be greater than 0", style);
    }
    if (t_size_cutoff > size_cutoff) {
      error->all(FLERR,
                 "size_cutoff for compute {} cannot be greater than it of compute cluster/size",
                 style);
    }
  }

  // Get ke/atom compute
  auto computes = lmp->modify->get_compute_by_style("pe/atom");
  if (computes.empty()) {
    error->all(FLERR, "compute {}: Cannot find compute with style 'pe/atom'", style);
  }
  compute_pe_atom = computes[0];

  size_local_rows = size_cutoff + 1;
  memory->create(local_pes, size_local_rows + 1, "compute:cluster/pe:local_pes");
  vector_local = local_pes;

  size_vector = size_cutoff + 1;
  memory->create(pes, size_vector + 1, "compute:cluster/pe:pes");
  vector = pes;
}

/* ---------------------------------------------------------------------- */

ComputeClusterPE::~ComputeClusterPE() noexcept(true)
{
  if (local_pes != nullptr) { memory->destroy(local_pes); }
  if (pes != nullptr) { memory->destroy(pes); }
}

/* ---------------------------------------------------------------------- */

void ComputeClusterPE::init()
{
  if ((modify->get_compute_by_style(style).size() > 1) && (comm->me == 0)) {
    error->warning(FLERR, "More than one compute {}", style);
  }
}

/* ---------------------------------------------------------------------- */

void ComputeClusterPE::compute_vector()
{
  invoked_vector = update->ntimestep;

  compute_local();

  ::memset(pes, 0.0, size_vector * sizeof(double));
  ::MPI_Allreduce(local_pes, pes, size_vector, MPI_DOUBLE, MPI_SUM, world);

  const double *dist = compute_cluster_size->vector;
  for (int i = 0; i < size_vector; ++i) { pes[i] /= dist[i]; }
}

/* ---------------------------------------------------------------------- */

void ComputeClusterPE::compute_local()
{
  invoked_local = update->ntimestep;

  if (compute_cluster_size->invoked_vector != update->ntimestep) {
    compute_cluster_size->compute_vector();
  }

  if (compute_pe_atom->invoked_peratom != update->ntimestep) { compute_pe_atom->compute_peratom(); }

  const double *const peratompes = compute_pe_atom->vector_atom;
  ::memset(local_pes, 0.0, size_local_rows * sizeof(double));

  for (const auto &[size, vec] : compute_cluster_size->cIDs_by_size) {
    if (size < size_cutoff) {
      for (const tagint cid : vec) {
        for (const tagint pid : compute_cluster_size->atoms_by_cID[cid]) {
          local_pes[size] += peratompes[pid];
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeClusterPE::memory_usage()
{
  return static_cast<double>((size_local_rows + size_vector) * sizeof(double));
}
