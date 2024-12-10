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
    error->all(FLERR, "compute cluster/enthropy: Cannot find compute with style 'cluster/size' with id: {}", arg[3]);
  }

  // Get the critical size
  size_cutoff = utils::inumeric(FLERR, arg[4], true, lmp);
  if (size_cutoff < 1) { error->all(FLERR, "size_cutoff for compute cluster/enthropy must be greater than 0"); }

  // Get ke/atom compute
  auto ke_computes = lmp->modify->get_compute_by_style("ke/atom");
  if (ke_computes.empty()) { error->all(FLERR, "compute cluster/enthropy: Cannot find compute with style 'ke/atom'"); }
  compute_ke_atom  = ke_computes[0];

  // Get pe/atom compute
  auto pe_computes = lmp->modify->get_compute_by_style("pe/atom");
  if (pe_computes.empty()) { error->all(FLERR, "compute cluster/enthropy: Cannot find compute with style 'pe/atom'"); }
  compute_pe_atom = pe_computes[0];

  size_local_rows = size_cutoff + 1;
  memory->create(local_temp, size_local_rows + 1, "compute:cluster/enthropy:temp");
  vector_local = local_temp;

  size_vector  = size_cutoff + 1;
  memory->create(temp, size_vector + 1, "compute:cluster/enthropy:temp");
  vector = temp;
}

/* ---------------------------------------------------------------------- */

ComputeClusterEnthropy::~ComputeClusterEnthropy() noexcept(true)
{
  if (local_temp != nullptr) { memory->destroy(local_temp); }
  if (temp != nullptr) { memory->destroy(temp); }
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

  ::MPI_Allreduce(local_temp, temp, size_vector, MPI_DOUBLE, MPI_SUM, world);

  const double* dist = compute_cluster_size->vector;
  for (int i = 0; i < size_vector; ++i) { temp[i] /= (dist[i] * i - 1) * domain->dimension; }
}

/* ---------------------------------------------------------------------- */

void ComputeClusterEnthropy::compute_local()
{
  invoked_local = update->ntimestep;

  if (compute_cluster_size->invoked_vector != update->ntimestep) { compute_cluster_size->compute_vector(); }

  if (compute_ke_atom->invoked_peratom != update->ntimestep) { compute_ke_atom->compute_peratom(); }

  const double* kes = compute_ke_atom->vector_atom;
  ::memset(local_temp, 0.0, size_local_rows * sizeof(double));

  // for (const auto &[size, vec] : compute_cluster_size->cIDs_by_size) {
  //   if (size < size_cutoff) {
  //     for (const tagint cid : vec) {
  //       for (const tagint pid : compute_cluster_size->atoms_by_cID[cid]) {
  //         local_temp[size] += 2 * kes[pid];
  //       }
  //     }
  //   }
  // }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeClusterEnthropy::memory_usage()
{
  return static_cast<double>((size_local_rows + size_vector) * sizeof(double));
}
