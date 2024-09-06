/*
 ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "fix_supersaturation_volume.h"
#include "compute.h"
#include "compute_supersaturation_mono.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "fmt/core.h"
#include "modify.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixSupersaturationVolume::FixSupersaturationVolume(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), screenflag(1), fileflag(0), next_step(0)
{

  restart_pbc = 1;
  nevery = 1;

  if (narg < 6) { utils::missing_cmd_args(FLERR, "fix supersaturation", error); }

  // Parse arguments //

  // Get compute supersaturation/mono
  compute_supersaturation_mono =
      dynamic_cast<ComputeSupersaturationMono *>(modify->get_compute_by_id(arg[3]));

  if (compute_supersaturation_mono == nullptr) {
    error->all(FLERR,
               "fix supersaturation: cannot find compute of style 'supersaturation/mono' with "
               "given id: {}",
               arg[4]);
  }

  // Get needed supersaturation
  supersaturation = utils::numeric(FLERR, arg[4], true, lmp);
  if (supersaturation <= 0) {
    error->all(FLERR, "Supersaturation for fix supersaturation must be positive");
  }

  // Get dampfing parameter
  damp = utils::numeric(FLERR, arg[5], true, lmp);
  if (damp <= 0 || damp > 1) {
    error->all(FLERR, "Dampfing parameter for fix supersaturation must be in range (0,1]");
  }
  damp = std::pow(damp, 1 / 3);

  // Parse optional keywords

  int iarg = 6;
  fp = nullptr;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "noscreen") == 0) {

      // Do not output to screen
      screenflag = 0;
      iarg += 1;

    } else if (strcmp(arg[iarg], "file") == 0) {

      // Write output to new file
      if (comm->me == 0) {
        fileflag = 1;
        fp = fopen(arg[iarg + 1], "w");
        if (fp == nullptr) {
          error->one(FLERR, "Cannot open fix supersaturation stats file {}: {}", arg[iarg + 1],
                     utils::getsyserror());
        }
      }
      iarg += 2;

    } else if (strcmp(arg[iarg], "append") == 0) {

      // Append output to file
      if (comm->me == 0) {
        fileflag = 1;
        fp = fopen(arg[iarg + 1], "a");
        if (fp == nullptr) {
          error->one(FLERR, "Cannot open fix supersaturation stats file {}: {}", arg[iarg + 1],
                     utils::getsyserror());
        }
      }
      iarg += 2;

    } else if (strcmp(arg[iarg], "nevery") == 0) {

      // Get execution period
      nevery = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg], "offset") == 0) {

      // Get start offset
      start_offset = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      if (start_offset < 0) {
        error->all(FLERR, "start_offset for fix supersaturation cannot be less than 0");
      }
      offflag = 1;
      iarg += 2;

    } else {
      error->all(FLERR, "Illegal fix supersaturation command option {}", arg[iarg]);
    }
  }

  if (comm->me == 0 && (fileflag != 0)) {
    fmt::print(fp, "ntimestep,delta,Vb,Va,deltaV,ssb,ssa,delta\n");
    fflush(fp);
  }

  next_step = update->ntimestep - (update->ntimestep % nevery);
  if (offflag != 0) { next_step = update->ntimestep + start_offset; }
}

/* ---------------------------------------------------------------------- */

FixSupersaturationVolume::~FixSupersaturationVolume() noexcept(true)
{
  if ((fp != nullptr) && (comm->me == 0)) {
    fflush(fp);
    fclose(fp);
  }
}

/* ---------------------------------------------------------------------- */

void FixSupersaturationVolume::init()
{
  // detect if any rigid fixes exist so rigid bodies can be rescaled
  // rfix[] = vector with pointers to each fix rigid

  rfix.clear();

  for (const auto &ifix : modify->get_fix_list()) {
    if (ifix->rigid_flag != 0) { rfix.push_back(ifix); }
  }
  if (comm->me == 0) {
    fmt::print(fp, "initialized\n");
    fflush(fp);
  }
}

/* ---------------------------------------------------------------------- */

int FixSupersaturationVolume::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSupersaturationVolume::end_of_step()
{
  if (comm->me == 0) {
    fmt::print(fp, "endoffff\n");
    fflush(fp);
  }
  if (update->ntimestep < next_step) { return; }
  next_step = update->ntimestep + nevery;

  if (compute_supersaturation_mono->invoked_scalar != update->ntimestep) {
    compute_supersaturation_mono->compute_scalar();
  }
  const double previous_supersaturation = compute_supersaturation_mono->scalar;
  const double volume_before = domain->volume();

  const auto delta = static_cast<double>(
      damp *
      (std::pow(static_cast<long double>(compute_supersaturation_mono->global_monomers) /
                    (compute_supersaturation_mono->execute_func() * supersaturation),
                1 / 3) -
       std::pow(domain->volume(), 1 / 3)) /
      2);

  if (comm->me == 0){
    fmt::print(fp, "delta: {}\n", delta);
    fflush(fp);
  }

  remap_before();

  if (comm->me == 0){
    fmt::print(fp, "remap before\n");
    fflush(fp);
  }

  // reset global and local box to new size/shape

  domain->boxlo[0] -= delta;
  domain->boxhi[0] += delta;
  domain->boxlo[1] -= delta;
  domain->boxhi[1] += delta;
  domain->boxlo[2] -= delta;
  domain->boxhi[2] += delta;

  if (comm->me == 0){
    fmt::print(fp, "Delted\n");
    fflush(fp);
  }

  domain->set_global_box();

  if (comm->me == 0){
    fmt::print(fp, "Global box\n");
    fflush(fp);
  }

  domain->set_local_box();

  if (comm->me == 0){
    fmt::print(fp, "Local box\n");
    fflush(fp);
  }

  remap_after();

  if (comm->me == 0){
    fmt::print(fp, "Remap after\n");
    fflush(fp);
  }

  const double volume_after = domain->volume();
  const double ssa = compute_supersaturation_mono->compute_scalar();

  if (comm->me == 0) {
    if (screenflag != 0) { utils::logmesg(lmp, "fix ss/volume: {:.5f}", delta); }
    if (fileflag != 0) {
      fmt::print(fp, "{},{:.5f},{:.5f},{:.5f},{:.5f},{:.3f},{:.3f},{:.3f}\n", update->ntimestep,
                 delta, volume_before, volume_after, volume_after - volume_before,
                 previous_supersaturation, ssa, ssa - previous_supersaturation);
      fflush(fp);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixSupersaturationVolume::remap_before() noexcept(true)
{
  // convert atoms and rigid bodies to lamda coords

  double **x = atom->x;
  const int *mask = atom->mask;
  int const nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if ((mask[i] & groupbit) != 0) { domain->x2lamda(x[i], x[i]); }
  }

  for (auto &ifix : rfix) { ifix->deform(0); }
}

/* ---------------------------------------------------------------------- */

void FixSupersaturationVolume::remap_after() noexcept(true)
{
  // convert atoms and rigid bodies back to box coords

  double **x = atom->x;
  const int *mask = atom->mask;
  int const nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if ((mask[i] & groupbit) != 0) { domain->lamda2x(x[i], x[i]); }
  }

  for (auto &ifix : rfix) { ifix->deform(1); }
}

/* ---------------------------------------------------------------------- */