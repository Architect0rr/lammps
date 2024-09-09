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
#include "atom_vec.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "fmt/core.h"
#include "irregular.h"
#include "modify.h"
#include "update.h"
#include "neigh_list.h"
#include "neighbor.h"

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
  box_change = BOX_CHANGE_ANY;

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
  need_exchange = false;
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
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);

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

void FixSupersaturationVolume::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

int FixSupersaturationVolume::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  // mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSupersaturationVolume::pre_exchange()
{
  if (update->ntimestep < next_step) { return; }
  next_step = update->ntimestep + nevery;
  need_exchange = true;
  force_reneighbor = update->ntimestep + 1;

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

  // domain->set_global_box();
  // domain->set_local_box();
  domain->reset_box();

  remap_after();

  neighbor->reset_timestep(update->ntimestep);


  if (domain->triclinic != 0) { domain->x2lamda(atom->nlocal); }
  domain->reset_box();
  if (domain->triclinic != 0) { domain->lamda2x(atom->nlocal); }

  bigint _nlocal = atom->nlocal;

  if (domain->triclinic != 0) { domain->x2lamda(atom->nlocal); }
  domain->reset_box();
  auto *irregular = new Irregular(lmp);
  irregular->migrate_atoms(1);
  delete irregular;
  if (domain->triclinic != 0) { domain->lamda2x(atom->nlocal); }

  bigint delta_atom = _nlocal - atom->nlocal;
  _nlocal = delta_atom > 0 ? delta_atom : 0;
  MPI_Allreduce(&_nlocal, &delta_atom, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  if (comm->me == 0){
    fmt::print(fp, "Transferred: {} atoms\n", delta_atom);
    fflush(fp);
  }

  // neighbor->build_one(list);

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

// void FixSupersaturationVolume::pre_exchange() {
//   if (!need_exchange) {return;}
//   need_exchange = false;

//   if (domain->triclinic != 0) { domain->x2lamda(atom->nlocal); }
//   domain->reset_box();
//   if (domain->triclinic != 0) { domain->lamda2x(atom->nlocal); }

//   bigint _nlocal = atom->nlocal;

//   if (domain->triclinic != 0) { domain->x2lamda(atom->nlocal); }
//   domain->reset_box();
//   auto *irregular = new Irregular(lmp);
//   irregular->migrate_atoms(1);
//   delete irregular;
//   if (domain->triclinic != 0) { domain->lamda2x(atom->nlocal); }

//   bigint delta_atom = _nlocal - atom->nlocal;
//   _nlocal = delta_atom > 0 ? delta_atom : 0;
//   MPI_Allreduce(&_nlocal, &delta_atom, 1, MPI_LMP_BIGINT, MPI_SUM, world);
//   if (comm->me == 0){
//     fmt::print(fp, "Transferred: {} atoms\n", delta_atom);
//     fflush(fp);
//   }
// }

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

void FixSupersaturationVolume::calculate_out(){
  bigint out_of_box = 0;
  double **x = atom->x;
  for (int i = 0; i < atom->nlocal; i++) {
    if (x[i][0] < domain->boxlo[0] || x[i][0] > domain->boxhi[0] || x[i][1] < domain->boxlo[1] || x[i][1] > domain->boxhi[1] || x[i][2] < domain->boxlo[2] || x[i][2] > domain->boxhi[2]) ++out_of_box;
  }

  bigint total_out = 0;
  MPI_Allreduce(&out_of_box, &total_out, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  if (comm->me == 0){
    fmt::print(fp, "Out of box: {}\n", total_out);
    fflush(fp);
  }
}

/* ---------------------------------------------------------------------- */

void FixSupersaturationVolume::print_box(){
    if (comm->me != 0) return;

  // Output all members of Domain class, excluding those derived from Pointers
  utils::logmesg(lmp, "box_exist: {} \n", domain->box_exist);
  utils::logmesg(lmp, "dimension: {} \n", domain->dimension);
  utils::logmesg(lmp, "nonperiodic: {} \n", domain->nonperiodic);
  utils::logmesg(lmp, "xperiodic: {} \n", domain->xperiodic);
  utils::logmesg(lmp, "yperiodic: {} \n", domain->yperiodic);
  utils::logmesg(lmp, "zperiodic: {} \n", domain->zperiodic);
  for (int i = 0; i < 3; ++i) {
      utils::logmesg(lmp, "periodicity[{}]: {} \n", i, domain->periodicity[i]);
      for (int j = 0; j < 2; ++j) {
      utils::logmesg(lmp, "boundary[{}][{}]: {} \n", i, j, domain->boundary[i][j]);
      }
  }
  utils::logmesg(lmp, "triclinic: {} \n", domain->triclinic);
  utils::logmesg(lmp, "triclinic_general: {} \n", domain->triclinic_general);
  utils::logmesg(lmp, "xprd: {} \n", domain->xprd);
  utils::logmesg(lmp, "yprd: {} \n", domain->yprd);
  utils::logmesg(lmp, "zprd: {} \n", domain->zprd);
  utils::logmesg(lmp, "xprd_half: {} \n", domain->xprd_half);
  utils::logmesg(lmp, "yprd_half: {} \n", domain->yprd_half);
  utils::logmesg(lmp, "zprd_half: {} \n", domain->zprd_half);
  for (int i = 0; i < 3; ++i) {
      utils::logmesg(lmp, "prd[{}]: {} \n", i, domain->prd[i]);
      utils::logmesg(lmp, "prd_half[{}]: {} \n", i, domain->prd_half[i]);
      utils::logmesg(lmp, "prd_lamda[{}]: {} \n", i, domain->prd_lamda[i]);
      utils::logmesg(lmp, "prd_half_lamda[{}]: {} \n", i, domain->prd_half_lamda[i]);
  }
  for (int i = 0; i < 3; ++i) {
      utils::logmesg(lmp, "boxlo[{}]: {} \n", i, domain->boxlo[i]);
      utils::logmesg(lmp, "boxhi[{}]: {} \n", i, domain->boxhi[i]);
      utils::logmesg(lmp, "boxlo_lamda[{}]: {} \n", i, domain->boxlo_lamda[i]);
      utils::logmesg(lmp, "boxhi_lamda[{}]: {} \n", i, domain->boxhi_lamda[i]);
      utils::logmesg(lmp, "boxlo_bound[{}]: {} \n", i, domain->boxlo_bound[i]);
      utils::logmesg(lmp, "boxhi_bound[{}]: {} \n", i, domain->boxhi_bound[i]);
  }
  for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 3; ++j) {
      utils::logmesg(lmp, "corners[{}][{}]: {} \n", i, j, domain->corners[i][j]);
      }
  }
  utils::logmesg(lmp, "minxlo: {} \n", domain->minxlo);
  utils::logmesg(lmp, "minxhi: {} \n", domain->minxhi);
  utils::logmesg(lmp, "minylo: {} \n", domain->minylo);
  utils::logmesg(lmp, "minyhi: {} \n", domain->minyhi);
  utils::logmesg(lmp, "minzlo: {} \n", domain->minzlo);
  utils::logmesg(lmp, "minzhi: {} \n", domain->minzhi);
  for (int i = 0; i < 3; ++i) {
      utils::logmesg(lmp, "sublo[{}]: {} \n", i, domain->sublo[i]);
      utils::logmesg(lmp, "subhi[{}]: {} \n", i, domain->subhi[i]);
      utils::logmesg(lmp, "sublo_lamda[{}]: {} \n", i, domain->sublo_lamda[i]);
      utils::logmesg(lmp, "subhi_lamda[{}]: {} \n", i, domain->subhi_lamda[i]);
  }
  utils::logmesg(lmp, "xy: {} \n", domain->xy);
  utils::logmesg(lmp, "xz: {} \n", domain->xz);
  utils::logmesg(lmp, "yz: {} \n", domain->yz);
  for (int i = 0; i < 6; ++i) {
      utils::logmesg(lmp, "h[{}]: {} \n", i, domain->h[i]);
      utils::logmesg(lmp, "h_inv[{}]: {} \n", i, domain->h_inv[i]);
      utils::logmesg(lmp, "h_rate[{}]: {} \n", i, domain->h_rate[i]);
  }
  for (int i = 0; i < 3; ++i) {
      utils::logmesg(lmp, "h_ratelo[{}]: {} \n", i, domain->h_ratelo[i]);
  }
  for (int i = 0; i < 3; ++i) {
      utils::logmesg(lmp, "avec[{}]: {} \n", i, domain->avec[i]);
      utils::logmesg(lmp, "bvec[{}]: {} \n", i, domain->bvec[i]);
      utils::logmesg(lmp, "cvec[{}]: {} \n", i, domain->cvec[i]);
  }
  for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
      utils::logmesg(lmp, "rotate_g2r[{}][{}]: {} \n", i, j, domain->rotate_g2r[i][j]);
      utils::logmesg(lmp, "rotate_r2g[{}][{}]: {} \n", i, j, domain->rotate_r2g[i][j]);
      }
  }
  utils::logmesg(lmp, "box_change: {} \n", domain->box_change);
  utils::logmesg(lmp, "box_change_size: {} \n", domain->box_change_size);
  utils::logmesg(lmp, "box_change_shape: {} \n", domain->box_change_shape);
  utils::logmesg(lmp, "box_change_domain: {} \n", domain->box_change_domain);
  utils::logmesg(lmp, "deform_flag: {} \n", domain->deform_flag);
  utils::logmesg(lmp, "deform_vremap: {} \n", domain->deform_vremap);
  utils::logmesg(lmp, "deform_groupbit: {} \n", domain->deform_groupbit);
  utils::logmesg(lmp, "copymode: {} \n", domain->copymode);

  double prd_half_lam[3];
  domain->x2lamda(domain->prd_half, prd_half_lam);
  double prd_half[3];
  domain->lamda2x(prd_half_lam, prd_half);
  utils::logmesg(lmp, "x_h: {} -x2l-> {} -l2x-> {} \n", domain->xprd_half, prd_half_lam[0], prd_half[0]);
  utils::logmesg(lmp, "y_h: {} -x2l-> {} -l2x-> {} \n", domain->yprd_half, prd_half_lam[1], prd_half[1]);
  utils::logmesg(lmp, "z_h: {} -x2l-> {} -l2x-> {} \n", domain->zprd_half, prd_half_lam[2], prd_half[2]);
}

/* ---------------------------------------------------------------------- */
