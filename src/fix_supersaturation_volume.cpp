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
#include "irregular.h"
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

  no_change_box = 1;
  restart_pbc = 1;
  pre_exchange_migrate = 1;
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
  damp = std::pow<double, double>(damp, 0.33333333333333);

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
  // if (comm->me == 0) {
  //   fmt::print(fp, "initialized\n");
  //   fflush(fp);
  // }
}

/* ---------------------------------------------------------------------- */

int FixSupersaturationVolume::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSupersaturationVolume::pre_exchange()
{
  if (update->ntimestep < next_step) { return; }
  next_step = update->ntimestep + nevery;

  // if (comm->me == 0) {
  //   fmt::print(fp, "endoffff\n");
  //   fflush(fp);
  // }

  if (compute_supersaturation_mono->invoked_scalar != update->ntimestep) {
    compute_supersaturation_mono->compute_scalar();
  }
  const double previous_supersaturation = compute_supersaturation_mono->scalar;
  const double volume_before = domain->volume();

  const auto global_monomers = static_cast<long double>(compute_supersaturation_mono->global_monomers);
  const long double ns1s = compute_supersaturation_mono->execute_func() * supersaturation;
  const long double needed_volume = global_monomers / ns1s;
  const long double needed_length = std::pow<long double, long double>(needed_volume, 0.33333333333333);
  const long double currenth_length = std::pow<long double, long double>(volume_before, 0.33333333333333);
  const auto delta = static_cast<double>(damp * (needed_length - currenth_length) / 2);


  if (comm->me == 0) {
    fmt::print(fp, "damp: {}, global_mono: {}, ns1s: {}, need_V: {}, need_L: {}, curr_V: {} curr_L: {}, delta: {}\n", damp, global_monomers, ns1s, needed_volume, needed_length, volume_before, currenth_length, delta);
    fflush(fp);
  }

  remap_before();

  // reset global and local box to new size/shape

  domain->boxlo[0] -= delta;
  domain->boxhi[0] += delta;
  domain->boxlo[1] -= delta;
  domain->boxhi[1] += delta;
  domain->boxlo[2] -= delta;
  domain->boxhi[2] += delta;

  remap_after();

  // domain->set_global_box();
  // domain->set_local_box();

  if (domain->triclinic != 0) { domain->x2lamda(atom->nlocal); }
  domain->reset_box();
  if (domain->triclinic != 0) { domain->lamda2x(atom->nlocal); }

  for (int i = 0; i < atom->nlocal; i++) { domain->remap(atom->x[i], atom->image[i]); }

  if (domain->triclinic != 0) { domain->x2lamda(atom->nlocal); }
  domain->reset_box();
  auto *irregular = new Irregular(lmp);
  irregular->migrate_atoms(1);
  delete irregular;
  if (domain->triclinic != 0) { domain->lamda2x(atom->nlocal); }

  //   domain->reset_box();

  // if (comm->me == 0) {
  //   fmt::print(fp, "Reset box\n");
  //   fflush(fp);
  // }

  // if (comm->me == 0){
  //   fmt::print(fp, "Remap after\n");
  //   fflush(fp);
  // }

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