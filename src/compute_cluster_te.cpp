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

#include "compute_cluster_te.h"
#include "compute_cluster_ke.h"
#include "compute_cluster_pe.h"
#include "compute_cluster_size.h"
#include "nucc_cspan.hpp"

#include "comm.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;
using NUCC::cspan;

/* ---------------------------------------------------------------------- */

ComputeClusterTE::ComputeClusterTE(LAMMPS *lmp, int narg, char **arg)
    : Compute(lmp, narg, arg) {
  vector_flag = 1;
  size_vector = 0;
  extvector = 0;
  local_flag = 1;
  size_local_rows = 0;
  size_local_cols = 0;

  if (narg < 3) {
    utils::missing_cmd_args(FLERR, "compute cluster/pe", error);
  }

  // Parse arguments //

  // Get cluster/size compute
  compute_cluster_size = dynamic_cast<ComputeClusterSize *>(
      lmp->modify->get_compute_by_id(arg[3]));
  if (compute_cluster_size == nullptr) {
    error->all(
        FLERR,
        "compute {}: Cannot find compute with style 'cluster/size' with id: {}",
        style, arg[3]);
  }

  // Get the critical size
  size_cutoff = compute_cluster_size->get_size_cutoff();
  if ((narg >= 4) && (::strcmp(arg[4], "inherit") != 0)) {
    int t_size_cutoff = utils::inumeric(FLERR, arg[4], true, lmp);
    if (t_size_cutoff < 1) {
      error->all(FLERR, "size_cutoff for compute {} must be greater than 0",
                 style);
    }
    if (t_size_cutoff > size_cutoff) {
      error->all(FLERR,
                 "size_cutoff for compute {} cannot be greater than it of "
                 "compute cluster/size",
                 style);
    }
  }

  // Get ke/atom compute
  auto cpe_computes = lmp->modify->get_compute_by_style("cluster/pe");
  if (cpe_computes.empty()) {
    error->all(FLERR, "compute {}: Cannot find compute with style 'cluster/pe'",
               style);
  }
  compute_cluster_pe = dynamic_cast<ComputeClusterPE *>(cpe_computes[0]);

  auto cke_computes = lmp->modify->get_compute_by_style("cluster/ke");
  if (cke_computes.empty()) {
    error->all(FLERR, "compute {}: Cannot find compute with style 'cluster/ke'",
               style);
  }
  compute_cluster_ke = dynamic_cast<ComputeClusterKE *>(cke_computes[0]);
}

/* ---------------------------------------------------------------------- */

ComputeClusterTE::~ComputeClusterTE() noexcept(true) {
  local_tes.destroy(memory);
  tes.destroy(memory);
}

/* ---------------------------------------------------------------------- */

void ComputeClusterTE::init() {
  if ((modify->get_compute_by_style(style).size() > 1) && (comm->me == 0)) {
    error->warning(FLERR, "More than one compute {}", style);
  }

  size_local_rows = size_cutoff + 1;
  local_tes.create(memory, size_local_rows, "compute:cluster/pe:local_pes");
  vector_local = local_tes.data();

  size_vector = size_cutoff + 1;
  tes.create(memory, size_vector, "compute:cluster/pe:pes");
  vector = tes.data();
}

/* ---------------------------------------------------------------------- */

void ComputeClusterTE::compute_vector() {
  invoked_vector = update->ntimestep;

  compute_local();

  tes.reset();
  ::MPI_Allreduce(local_tes.data(), tes.data(), size_vector, MPI_DOUBLE,
                  MPI_SUM, world);
}

/* ---------------------------------------------------------------------- */

void ComputeClusterTE::compute_local() {
  invoked_local = update->ntimestep;

  if (compute_cluster_ke->invoked_vector != update->ntimestep) {
    compute_cluster_ke->compute_vector();
  }
  if (compute_cluster_pe->invoked_vector != update->ntimestep) {
    compute_cluster_pe->compute_vector();
  }

  const cspan<const double> per_cluster_pes =
      compute_cluster_ke->get_data_local();
  const cspan<const double> per_cluster_kes =
      compute_cluster_pe->get_data_local();
  local_tes.reset();

  for (int i = 0; i < size_vector; i++) {
    local_tes[i] = per_cluster_pes[i] + per_cluster_kes[i];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeClusterTE::memory_usage() {
  return static_cast<double>(tes.memory_usage() + local_tes.memory_usage());
}
