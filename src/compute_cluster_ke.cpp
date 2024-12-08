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

#include "compute_cluster_ke.h"
#include "compute_cluster_size.h"
#include "nucc_cspan.hpp"

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

ComputeClusterKE::ComputeClusterKE(LAMMPS *lmp, int narg, char **arg)
    : Compute(lmp, narg, arg) /*, substract_vcm(0)*/
{
  vector_flag = 1;
  size_vector = 0;
  extvector = 0;
  local_flag = 1;
  size_local_rows = 0;
  size_local_cols = 0;

  if (narg < 4) {
    utils::missing_cmd_args(FLERR, "compute cluster/ke", error);
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

  size_cutoff = compute_cluster_size->get_size_cutoff();

  int iarg = 4;
  while (iarg < narg) {
    if (iarg + 2 > narg) {
      error->all(FLERR, "Illegal compute cluster/ke command");
    }
    if (::strcmp(arg[iarg], "cut") == 0) {
      int t_size_cutoff = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
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
      iarg += 2;
      // } else if (::strcmp(arg[iarg], "substract_vcm") == 0) {
      //   substract_vcm = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      //   iarg += 2;
    } else {
      error->all(FLERR, "Illegal fix langevin command");
    }
  }

  // double vcm[3]{};
  // double masstotal = group->mass(igroup);
  // group->vcm(igroup, masstotal, vcm);

  // Get ke/atom compute
  auto computes = lmp->modify->get_compute_by_style("ke/atom");
  if (computes.empty()) {
    error->all(FLERR, "compute {}: Cannot find compute with style 'ke/atom'",
               style);
  }
  compute_ke_atom = computes[0];
}

/* ---------------------------------------------------------------------- */

ComputeClusterKE::~ComputeClusterKE() noexcept(true) {
  local_kes.destroy(memory);
  kes.destroy(memory);
}

/* ---------------------------------------------------------------------- */

void ComputeClusterKE::init() {
  if ((modify->get_compute_by_style(style).size() > 1) && (comm->me == 0)) {
    error->warning(FLERR, "More than one compute {}", style);
  }

  size_local_rows = size_cutoff + 1;
  local_kes.create(memory, size_local_rows, "compute:cluster/ke:local_kes");
  vector_local = local_kes.data();

  size_vector = size_cutoff + 1;
  kes.create(memory, size_vector, "compute:cluster/ke:kes");
  vector = kes.data();
}

/* ---------------------------------------------------------------------- */

void ComputeClusterKE::compute_vector() {
  invoked_vector = update->ntimestep;

  compute_local();

  kes.reset();
  ::MPI_Allreduce(local_kes.data(), kes.data(), size_vector, MPI_DOUBLE,
                  MPI_SUM, world);

  const double *dist = compute_cluster_size->vector;
  for (int i = 0; i < size_vector; ++i) {
    if (dist[i] > 0) {
      kes[i] /= dist[i];
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeClusterKE::compute_local() {
  invoked_local = update->ntimestep;

  if (compute_cluster_size->invoked_vector != update->ntimestep) {
    compute_cluster_size->compute_vector();
  }

  if (compute_ke_atom->invoked_peratom != update->ntimestep) {
    compute_ke_atom->compute_peratom();
  }

  const double *const peratomkes = compute_ke_atom->vector_atom;
  local_kes.reset();

  for (const auto &[size, vec] : compute_cluster_size->cIDs_by_size) {
    if (size < size_cutoff) {
      for (const tagint cid : vec) {
        for (const tagint pid : compute_cluster_size->atoms_by_cID[cid]) {
          local_kes[size] += peratomkes[pid];
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeClusterKE::memory_usage() {
  return static_cast<double>(kes.memory_usage() + local_kes.memory_usage());
}
