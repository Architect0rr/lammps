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

#include "compute_supersaturation_mono.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "modify.h"
#include "update.h"

#include <cmath>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSupersaturationMono::ComputeSupersaturationMono(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), local_scalar(0), local_monomers(0)
{

  scalar_flag = 1;
  extscalar = 0;
  local_flag = 1;

  if (narg < 7) utils::missing_cmd_args(FLERR, "compute supersaturation", error);

  // Parse arguments //

  // Target region
  region = domain->get_region_by_id(arg[3]);
  if (region == nullptr) {
    error->all(FLERR, "compute supersaturation: Cannot find target region {}", arg[3]);
  }

  // Get neighs compute
  compute_neighs = lmp->modify->get_compute_by_id(arg[4]);
  if (compute_neighs == nullptr) {
    error->all(FLERR,
               "compute supersaturation: Cannot find compute with style 'coord/atom' with id: {}",
               arg[4]);
  }

  auto temp_computes = lmp->modify->get_compute_by_style("temp");
  if (temp_computes.size() == 0) {
    error->all(FLERR, "compute supersaturation: Cannot find compute with style 'temp'.");
  }
  compute_temp = temp_computes[0];

  // Arrhenius coeffs
  coeffs[0] = utils::numeric(FLERR, arg[5], true, lmp);
  coeffs[1] = utils::numeric(FLERR, arg[6], true, lmp);
}

/* ---------------------------------------------------------------------- */

ComputeSupersaturationMono::~ComputeSupersaturationMono() {}

/* ---------------------------------------------------------------------- */

void ComputeSupersaturationMono::init()
{
  if (modify->get_compute_by_style(style).size() > 1)
    if (comm->me == 0) error->warning(FLERR, "More than one compute {}", style);
}

/* ---------------------------------------------------------------------- */

double ComputeSupersaturationMono::compute_scalar()
{
  invoked_scalar = update->ntimestep;

  compute_local();
  MPI_Allreduce(&local_monomers, &global_monomers, 1, MPI_LMP_BIGINT, MPI_SUM, world);

  scalar = static_cast<double>(global_monomers) / domain->volume() / execute_func();
  return scalar;
}

/* ---------------------------------------------------------------------- */

void ComputeSupersaturationMono::compute_local()
{
  invoked_local = update->ntimestep;

  local_monomers = 0;
  if (compute_neighs->invoked_peratom != update->ntimestep) { compute_neighs->compute_peratom(); }
  if (compute_temp->invoked_scalar != update->ntimestep) { compute_temp->compute_scalar(); }
  for (int i = 0; i < atom->nlocal; ++i) {
    if (atom->mask[i] & groupbit) {
      if (compute_neighs->vector_atom[i] == 0) { ++local_monomers; }
    }
  }

  local_scalar = static_cast<double>(local_monomers) / domain->subvolume() / execute_func();
}

/* ---------------------------------------------------------------------- */

double ComputeSupersaturationMono::execute_func()
{
  return coeffs[0] * exp(-coeffs[1] / compute_temp->scalar);
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeSupersaturationMono::memory_usage()
{
  double bytes = 0;
  return bytes;
}