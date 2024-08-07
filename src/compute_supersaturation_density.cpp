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

#include "compute_supersaturation_density.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <cmath>

using namespace LAMMPS_NS;

static constexpr double EPSILON = 1.0e-6;
/* ---------------------------------------------------------------------- */

ComputeSupersaturationDensity::ComputeSupersaturationDensity(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg)
{

  scalar_flag = 1;
  extscalar = 0;

  if (narg < 8) utils::missing_cmd_args(FLERR, "compute supersaturation", error);

  // Parse arguments //

  // Target region
  region = domain->get_region_by_id(arg[3]);
  if (region == nullptr){
    error->all(FLERR, "compute supersaturation: Cannot find target region {}", arg[3]);
  }

  // Get cluster/size compute
  compute_cluster_size = lmp->modify->get_compute_by_id(arg[4]);
  if (compute_cluster_size == nullptr){
    error->all(FLERR, "compute supersaturation: Cannot find compute with style 'cluster/size' with id: {}", arg[4]);
  }

  // Get kmax
  kmax = utils::inumeric(FLERR, arg[5], true, lmp);
  if (kmax < 1)
    error->all(FLERR, "kmax for compute supersaturation/density cannot be less than 1");

  // Arrhenius coeffs
  coeffs[0] = utils::numeric(FLERR, arg[6], true, lmp);
  coeffs[1] = utils::numeric(FLERR, arg[7], true, lmp);

  auto temp_computes = lmp->modify->get_compute_by_style("temp");
  if (temp_computes.size() == 0){
    error->all(FLERR, "compute supersaturation: Cannot find compute with style 'temp'.");
  }
  compute_temp = temp_computes[0];
}

/* ---------------------------------------------------------------------- */

ComputeSupersaturationDensity::~ComputeSupersaturationDensity() {}

/* ---------------------------------------------------------------------- */

void ComputeSupersaturationDensity::init()
{
  if (modify->get_compute_by_style(style).size() > 1)
    if (comm->me == 0) error->warning(FLERR, "More than one compute {}", style);
}

/* ---------------------------------------------------------------------- */

double ComputeSupersaturationDensity::compute_scalar() {
  invoked_scalar = update->ntimestep;

  if (compute_cluster_size->invoked_vector != update->ntimestep){
    compute_cluster_size->compute_vector();
  }
if (compute_temp->invoked_scalar != update->ntimestep){
    compute_temp->compute_scalar();
  }

  double *dist = compute_cluster_size->vector;
  double sum{};
  for (int i = 1; i<= kmax; ++i){
    sum += dist[i];
  }

  // for (int i = 0; i < compute_cluster_size->size_vector; ++i){
  //   sum += dist[i];
  // }

  scalar = sum / domain->volume() / execute_func();
  return scalar;
}

/* ---------------------------------------------------------------------- */

double ComputeSupersaturationDensity::execute_func()
{
  return coeffs[0]*exp(-coeffs[1]/compute_temp->scalar);
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeSupersaturationDensity::memory_usage()
{
  double bytes = 0;
  return bytes;
}
