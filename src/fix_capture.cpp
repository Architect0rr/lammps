/*
 ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "fix_capture.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "fmt/core.h"
#include "irregular.h"
#include "lattice.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <time.h>
#include <cmath>
#include <cstring>
#include <unordered_map>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixCapture::FixCapture(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{

  restart_pbc = 1;
  sigmas = nullptr;
  vmeans = nullptr;

  if (domain->dimension == 2) { error->all(FLERR, "cluster/crush is not compatible with 2D yet"); }

  if (narg < 6) utils::missing_cmd_args(FLERR, "cluster/crush", error);

  // Parse arguments //

  // Target region
  region = domain->get_region_by_id(arg[3]);
  if (region == nullptr) { error->all(FLERR, "Cannot find target region {}", arg[3]); }

  // Get the seed for velocity generator
  int vseed = utils::numeric(FLERR, arg[4], true, lmp);
  vrandom = new RanPark(lmp, vseed);

  // Get number of sigmas
  nsigma = utils::numeric(FLERR, arg[5], true, lmp);

  // Get temp compute
  auto temp_computes = lmp->modify->get_compute_by_style("temp");
  if (temp_computes.size() == 0) {
    error->all(FLERR, "compute supersaturation: Cannot find compute with style 'temp'.");
  }
  compute_temp = temp_computes[0];

  int triclinic = domain->triclinic;

  // bounding box for atom creation
  // only limit bbox by region if its bboxflag is set (interior region)

  if (triclinic == 0) {
    xlo = domain->boxlo[0];
    xhi = domain->boxhi[0];
    ylo = domain->boxlo[1];
    yhi = domain->boxhi[1];
    zlo = domain->boxlo[2];
    zhi = domain->boxhi[2];
  } else {
    xlo = domain->boxlo_bound[0];
    xhi = domain->boxhi_bound[0];
    ylo = domain->boxlo_bound[1];
    yhi = domain->boxhi_bound[1];
    zlo = domain->boxlo_bound[2];
    zhi = domain->boxhi_bound[2];
    boxlo = domain->boxlo_lamda;
    boxhi = domain->boxhi_lamda;
  }

  if (region && region->bboxflag) {
    xlo = MAX(xlo, region->extent_xlo);
    xhi = MIN(xhi, region->extent_xhi);
    ylo = MAX(ylo, region->extent_ylo);
    yhi = MIN(yhi, region->extent_yhi);
    zlo = MAX(zlo, region->extent_zlo);
    zhi = MIN(zhi, region->extent_zhi);
  }

  if (xlo > xhi || ylo > yhi || zlo > zhi)
    error->all(FLERR, "No overlap of box and region for cluster/crush");
}

/* ---------------------------------------------------------------------- */

FixCapture::~FixCapture()
{
  delete vrandom;
  if (sigmas != nullptr){
    memory->destroy(sigmas);
  }
}

/* ---------------------------------------------------------------------- */

int FixCapture::setmask()
{
  int mask = 0;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCapture::init() {
  if (sigmas != nullptr){
    memory->destroy(sigmas);
  }
  memory->create(sigmas, atom->ntypes, "fix_capture:sigmas");
  if (vmeans != nullptr){
    memory->destroy(vmeans);
  }
  memory->create(vmeans, atom->ntypes, "fix_capture:vmeans");
}

/* ---------------------------------------------------------------------- */

void FixCapture::final_integrate()
{
  if (compute_temp->invoked_scalar != update->ntimestep){
    compute_temp->compute_scalar();
  }

  constexpr long double c_v = 0.7978845608028653558798921198687L;    // sqrt(2/pi)
  for (int i = 0; i < atom->ntypes; ++i){
    sigmas[i] = std::sqrt(compute_temp->scalar / atom->mass[i]);
    vmeans[i] = c_v * sigmas[i];
  }

  //
  //     double sigma = std::sqrt(monomer_temperature / atom->mass[atom->type[pID]]);
  //     double v_mean = c_v * sigma;
  //     v[pID][0] = v_mean + vrandom->gaussian() * sigma;

  double **v = atom->v;

  for (int i = 0; i < atom->nlocal; ++i){
    int atype = atom->type[i];
    if (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2] > vmeans[atype] + nsigma * sigmas[atype]){
      v[i][0] = vmeans[atype] + vrandom->gaussian() * sigmas[atype];
      v[i][1] = vmeans[atype] + vrandom->gaussian() * sigmas[atype];
      v[i][2] = vmeans[atype] + vrandom->gaussian() * sigmas[atype];
    }
  }
}

/* ---------------------------------------------------------------------- */