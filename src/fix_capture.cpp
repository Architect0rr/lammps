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
  if (comm->me == 0){
    logfile = fopen("fix_capture.log", "a");
    if (logfile == nullptr)
      error->one(FLERR, "Cannot open fix capture log file {}: {}", "fix_capture.log", utils::getsyserror());
    fmt::print(logfile, "ts,n,vmean,sigma,mean_md\n");
    fflush(logfile);
  }
}

/* ---------------------------------------------------------------------- */

FixCapture::~FixCapture()
{
  delete vrandom;
  if (comm->me == 0){
    fflush(logfile);
    fclose(logfile);
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
  typeids.clear();
  typeids.reserve(atom->ntypes);
  for (int i = 0; i < atom->nlocal; ++i){
    typeids.emplace(atom->type[i], std::make_pair<double, double>(0.0, 0.0));
  }
  for (const auto& [k, v] : typeids){
    if (!atom->mass_setflag[k]){
      error->all(FLERR, "fix capture: mass is not set for atom type {}.", k);
    }
  }
  // if (atom->mass_setflag){
  //   error->all(FLERR, "fix capture: mass is not set.");
  // }
}

/* ---------------------------------------------------------------------- */

void FixCapture::final_integrate()
{
  if (compute_temp->invoked_scalar != update->ntimestep){
    compute_temp->compute_scalar();
  }

  // constexpr long double c_v = 0.7978845608028653558798921198687L;  // sqrt(2/pi)
  constexpr long double c_v = 1.4142135623730950488016887242097L;  // sqrt(2)
  for (auto& [k, v] : typeids){
    v.first = sqrt(compute_temp->scalar / atom->mass[k]);
    v.second = c_v * v.first;
  }

  double **v = atom->v;

  bigint ncaptured_local = 0;
  for (int i = 0; i < atom->nlocal; ++i){
    const auto& [sigma, vmean] = typeids[atom->type[i]];
    constexpr long double a_v = 1.7320508075688772935274463415059L;  // sqrt(3)
    if (sqrt(v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]) > /*a_v **/ (vmean + nsigma * sigma)){
      v[i][0] = (vmean + vrandom->gaussian() * sigma)/2;
      v[i][1] = (vmean + vrandom->gaussian() * sigma)/2;
      v[i][2] = (vmean + vrandom->gaussian() * sigma)/2;

      ++ncaptured_local;
    }
  }

  bigint ncaptured_total = 0;
  MPI_Allreduce(&ncaptured_local, &ncaptured_total, 1, MPI_LMP_BIGINT, MPI_SUM, world);

  constexpr long double a_v = 0.8*1.0220217810393767580226573302752L;
  constexpr long double b_v = 0.1546370863640482533333333333333L;
  double rl = a_v*exp(b_v*pow(compute_temp->scalar, 2.791206046910478));

  bigint nclose_local = 0;
  double **x = atom->x;
  for (int i = 0; i < atom->nlocal; ++i){
    for (int j = i + 1; j < atom->nghost; ++j){
      double dx, dy, dz;
      dx = x[i][0] - x[j][0];
      dy = x[i][1] - x[j][1];
      dz = x[i][2] - x[j][2];
      if (dx*dx + dy*dy + dz*dz < rl*rl){
        ++nclose_local;
      }
    }
  }

  bigint nclose_total = 0;
  MPI_Allreduce(&nclose_local, &nclose_total, 1, MPI_LMP_BIGINT, MPI_SUM, world);

  if (comm->me == 0){
    fmt::print(logfile, "{},{}, {}\n", update->ntimestep, ncaptured_total, nclose_total);
    fflush(logfile);
  }
}

/* ---------------------------------------------------------------------- */