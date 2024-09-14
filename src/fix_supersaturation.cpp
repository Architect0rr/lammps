/*
 ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "fix_supersaturation.h"
#include "compute.h"
#include "compute_supersaturation_mono.h"

#include "atom.h"
#include "atom_vec.h"
#include "atom_vec_body.h"
#include "atom_vec_ellipsoid.h"
#include "atom_vec_line.h"
#include "atom_vec_tri.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "fmt/core.h"
#include "lattice.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <bits/std_abs.h>
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

constexpr int DEFAULT_MAXTRY = 1000;
constexpr int DEFAULT_MAXTRY_CALL = 5;
constexpr int DEFAULT_MAXTRY_MOVE = 9;

/* ---------------------------------------------------------------------- */

FixSupersaturation::FixSupersaturation(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), localflag(false), randomflag(true), moveflag(false), screenflag(1),
    fileflag(0), next_step(0), maxtry(::DEFAULT_MAXTRY), maxtry_call(::DEFAULT_MAXTRY_CALL),
    maxtry_move(::DEFAULT_MAXTRY_MOVE), scaleflag(0), fix_temp(0), offflag(0)
{

  nevery = 1;

  if (narg < 10) { utils::missing_cmd_args(FLERR, "fix supersaturation", error); }

  // Parse arguments //

  // Target region
  region = domain->get_region_by_id(arg[3]);
  if (region == nullptr) { error->all(FLERR, "Cannot find target region {}", arg[3]); }

  // Get compute supersaturation/mono
  compute_supersaturation_mono =
      dynamic_cast<ComputeSupersaturationMono *>(modify->get_compute_by_id(arg[4]));

  if (compute_supersaturation_mono == nullptr) {
    error->all(FLERR,
               "fix supersaturation: cannot find compute of style 'supersaturation/mono' with "
               "given id: {}",
               arg[4]);
  }

  // Minimum distance to other atoms from the place atom is inserted to
  overlap = utils::numeric(FLERR, arg[5], true, lmp);
  if (overlap < 0) {
    error->all(FLERR, "Minimum distance for fix supersaturation must be non-negative");
  }

  // apply scaling factor for styles that use distance-dependent factors
  overlap *= domain->lattice->xlattice;
  odistsq = overlap * overlap;

  // # of type of atoms to insert
  ntype = utils::inumeric(FLERR, arg[6], true, lmp);
  if ((ntype <= 0) || (ntype > atom->ntypes)) {
    error->all(FLERR, "Invalid atom type in create_atoms command");
  }

  // Get the seed for coordinate generator
  int const xseed = utils::inumeric(FLERR, arg[7], true, lmp);
  xrandom = new RanPark(lmp, xseed);

  // Get needed supersaturation
  supersaturation = utils::numeric(FLERR, arg[8], true, lmp);
  if (supersaturation <= 0) {
    error->all(FLERR, "Supersaturation for fix supersaturation must be positive");
  }

  // Get dampfing parameter
  damp = utils::numeric(FLERR, arg[9], true, lmp);
  if ((damp <= 0) || (damp > 1)) {
    error->all(FLERR, "Dampfing parameter for fix supersaturation must be in range (0,1]");
  }

  // Parse optional keywords

  int iarg = 10;
  fp = nullptr;

  while (iarg < narg) {
    if (::strcmp(arg[iarg], "maxtry") == 0) {

      // Max attempts to search for a new suitable location
      maxtry = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      if (maxtry < 1) { error->all(FLERR, "maxtry for fix supersaturation cannot be less than 1"); }
      iarg += 2;

    } else if (::strcmp(arg[iarg], "maxtry_move") == 0) {

      // Max attempts to search for a new suitable location
      maxtry_move = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      if (maxtry_move < 1) {
        error->all(FLERR, "maxtry_move for fix supersaturation cannot be less than 1");
      }
      iarg += 2;

    } else if (::strcmp(arg[iarg], "mode") == 0) {

      if (::strcmp(arg[iarg + 1], "local") == 0) {
        localflag = true;
      } else if (::strcmp(arg[iarg + 1], "universe") == 0) {
        localflag = false;
      } else {
        error->all(FLERR, "Unknown mode for fix supersaturation: {}", arg[iarg + 1]);
      }
      iarg += 2;

    } else if (::strcmp(arg[iarg], "method") == 0) {

      if (::strcmp(arg[iarg + 1], "random") == 0) {
        randomflag = true;
      } else if (::strcmp(arg[iarg + 1], "grid") == 0) {
        randomflag = false;
      } else {
        error->all(FLERR, "Unknown method for fix supersaturation: {}", arg[iarg + 1]);
      }
      iarg += 2;

    } else if (::strcmp(arg[iarg], "move") == 0) {

      if (::strcmp(arg[iarg + 1], "yes") == 0) {
        moveflag = true;
      } else if (::strcmp(arg[iarg + 1], "no") == 0) {
        moveflag = false;
      } else {
        error->all(FLERR, "Unknown move for fix supersaturation: {}", arg[iarg + 1]);
      }
      iarg += 2;

    } else if (::strcmp(arg[iarg], "maxtry_call") == 0) {

      // Get max number of tries for calling delete_monomers()/add_monomers()
      maxtry_call = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      if (maxtry_call < 1) {
        error->all(FLERR, "maxtry_call for fix supersaturation cannot be less than 1");
      }
      iarg += 2;

    } else if (::strcmp(arg[iarg], "temp") == 0) {

      // Monomer temperature
      fix_temp = 1;
      monomer_temperature = utils::numeric(FLERR, arg[iarg + 1], true, lmp);
      if (monomer_temperature < 0) {
        error->all(FLERR, "Monomer temperature for fix supersaturation cannot be negative");
      }

      // Get the seed for velocity generator
      int const vseed = utils::inumeric(FLERR, arg[iarg + 2], true, lmp);
      vrandom = new RanPark(lmp, vseed);
      iarg += 3;

    } else if (::strcmp(arg[iarg], "noscreen") == 0) {

      // Do not output to screen
      screenflag = 0;
      iarg += 1;

    } else if (::strcmp(arg[iarg], "file") == 0) {

      // Write output to new file
      if (comm->me == 0) {
        fileflag = 1;
        fp = ::fopen(arg[iarg + 1], "w");
        if (fp == nullptr) {
          error->one(FLERR, "Cannot open fix supersaturation stats file {}: {}", arg[iarg + 1],
                     utils::getsyserror());
        }
      }
      iarg += 2;

    } else if (::strcmp(arg[iarg], "append") == 0) {

      // Append output to file
      if (comm->me == 0) {
        fileflag = 1;
        fp = ::fopen(arg[iarg + 1], "a");
        if (fp == nullptr) {
          error->one(FLERR, "Cannot open fix supersaturation stats file {}: {}", arg[iarg + 1],
                     utils::getsyserror());
        }
      }
      iarg += 2;

    } else if (::strcmp(arg[iarg], "nevery") == 0) {

      // Get execution period
      nevery = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      iarg += 2;

    } else if (::strcmp(arg[iarg], "offset") == 0) {

      // Get start offset
      start_offset = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      if (start_offset < 0) {
        error->all(FLERR, "start_offset for fix supersaturation cannot be less than 0");
      }
      offflag = 1;
      iarg += 2;

    } else if (::strcmp(arg[iarg], "units") == 0) {

      if (::strcmp(arg[iarg + 1], "box") == 0) {
        scaleflag = 0;
      } else if (::strcmp(arg[iarg + 1], "lattice") == 0) {
        scaleflag = 1;
      } else {
        error->all(FLERR, "Unknown fix supersaturation units option {}", arg[iarg + 1]);
      }
      iarg += 2;

    } else {
      error->all(FLERR, "Illegal fix supersaturation command option {}", arg[iarg]);
    }
  }

  if ((!localflag) && ((!randomflag) || moveflag)) {
    error->all(
        FLERR,
        "fix supersaturation: mode UNIVERSE can be used only with method RANDOM and no moving",
        arg[iarg]);
  }

  triclinic = domain->triclinic;

  // bounding box for atom creation
  // only limit bbox by region if its bboxflag is set (interior region)

  if (triclinic == 0) {
    globbonds[0][0] = domain->boxlo[0];
    globbonds[0][1] = domain->boxhi[0];
    globbonds[1][0] = domain->boxlo[1];
    globbonds[1][1] = domain->boxhi[1];
    globbonds[2][0] = domain->boxlo[2];
    globbonds[2][1] = domain->boxhi[2];
    subbonds[0][0] = domain->sublo[0];
    subbonds[0][1] = domain->subhi[0];
    subbonds[1][0] = domain->sublo[1];
    subbonds[1][1] = domain->subhi[1];
    subbonds[2][0] = domain->sublo[2];
    subbonds[2][1] = domain->subhi[2];
  } else {
    globbonds[0][0] = domain->boxlo_bound[0];
    globbonds[0][1] = domain->boxhi_bound[0];
    globbonds[1][0] = domain->boxlo_bound[1];
    globbonds[1][1] = domain->boxhi_bound[1];
    globbonds[2][0] = domain->boxlo_bound[2];
    globbonds[2][1] = domain->boxhi_bound[2];
    subbonds[0][0] = domain->sublo_lamda[0];
    subbonds[0][1] = domain->subhi_lamda[0];
    subbonds[1][0] = domain->sublo_lamda[1];
    subbonds[1][1] = domain->subhi_lamda[1];
    subbonds[2][0] = domain->sublo_lamda[2];
    subbonds[2][1] = domain->subhi_lamda[2];
    boxlo = domain->boxlo_lamda;
    boxhi = domain->boxhi_lamda;
  }

  if ((region != nullptr) && (region->bboxflag != 0)) {
    globbonds[0][0] = MAX(globbonds[0][0], region->extent_xlo);
    globbonds[0][1] = MIN(globbonds[0][1], region->extent_xhi);
    globbonds[1][0] = MAX(globbonds[1][0], region->extent_ylo);
    globbonds[1][1] = MIN(globbonds[1][1], region->extent_yhi);
    globbonds[2][0] = MAX(globbonds[2][0], region->extent_zlo);
    globbonds[2][1] = MIN(globbonds[2][1], region->extent_zhi);
    subbonds[0][0] = MAX(subbonds[0][0], region->extent_xlo);
    subbonds[0][1] = MIN(subbonds[0][1], region->extent_xhi);
    subbonds[1][0] = MAX(subbonds[1][0], region->extent_ylo);
    subbonds[1][1] = MIN(subbonds[1][1], region->extent_yhi);
    subbonds[2][0] = MAX(subbonds[2][0], region->extent_zlo);
    subbonds[2][1] = MIN(subbonds[2][1], region->extent_zhi);
  }

  if ((globbonds[0][0] > globbonds[0][1]) || (globbonds[1][0] > globbonds[1][1]) ||
      (globbonds[2][0] > globbonds[2][1])) {
    error->all(FLERR, "No overlap of box and region for fix keep/count");
  }

  if ((comm->me == 0) && (fileflag != 0)) {
    fmt::print(fp, "ntimestep,ntotal,a2d,a2a,ad,aa,ssb,ssa,del,succrate\n");
    ::fflush(fp);
  }

  next_step = update->ntimestep - (update->ntimestep % nevery);
  if (offflag != 0) { next_step = update->ntimestep + start_offset; }

  memory->create(pproc, comm->nprocs * sizeof(int), "fix_supersaturation:pproc");

  ::memset(xone, 0, 3 * sizeof(double));
  ::memset(lamda, 0, 3 * sizeof(double));

  alogrand = new RanPark(lmp, comm->nprocs);
}

/* ---------------------------------------------------------------------- */

FixSupersaturation::~FixSupersaturation() noexcept(true)
{
  delete xrandom;
  delete vrandom;
  delete alogrand;

  if ((fp != nullptr) && (comm->me == 0)) {
    ::fflush(fp);
    ::fclose(fp);
  }
  memory->destroy(pproc);
}

/* ---------------------------------------------------------------------- */

void FixSupersaturation::init()
{
  if ((modify->get_fix_by_style(style).size() > 1) && (comm->me == 0)) {
    error->warning(FLERR, "More than one fix {}", style);
  }
}

/* ---------------------------------------------------------------------- */

int FixSupersaturation::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSupersaturation::pre_exchange()
{
  if (update->ntimestep < next_step) { return; }
  next_step = update->ntimestep + nevery;

  if (compute_supersaturation_mono->invoked_scalar != update->ntimestep) {
    compute_supersaturation_mono->compute_scalar();
  }
  const double previous_supersaturation = compute_supersaturation_mono->scalar;

  auto delta = static_cast<bigint>(
      std::floor(static_cast<long double>(compute_supersaturation_mono->execute_func() *
                                          domain->volume() * supersaturation) -
                 compute_supersaturation_mono->global_monomers));

  const bool delflag = delta < 0;
  delta = static_cast<bigint>(
      std::floor(static_cast<long double>(damp) * static_cast<long double>(std::abs(delta))));

  const bigint natoms_previous = atom->natoms;
  const int nlocal_previous = atom->nlocal;

  if (delta != 0) {
    if (!delflag) {
      // clear global->local map for owned and ghost atoms
      // clear ghost count and any ghost bonus data internal to AtomVec
      // same logic as beginning of Comm::exchange()
      // do it now b/c creating atoms will overwrite ghost atoms

      if (atom->map_style != Atom::MAP_NONE) { atom->map_clear(); }
      atom->nghost = 0;
      atom->avec->clear_bonus();

      region->prematch();
    }

    if ((!delflag) && (!localflag)) {
      add_monomers_universe_random(delta);
    } else {
      bigint sum = delta;
      int tries = maxtry_call;

      do {
        pproc[comm->me] = static_cast<int>(sum / comm->nprocs);
        if (static_cast<int>(alogrand->uniform() * 32767) == comm->me) {
          pproc[comm->me] += sum % comm->nprocs;
        }

        if (pproc[comm->me] > 0) {
          if (delflag) {
            delete_monomers();
          } else {
            if (randomflag) {
              add_monomers_local_random();
            } else {
              add_monomers_local_grid();
            }
          }
        }

        int temp = pproc[comm->me];
        ::memset(pproc, 0, comm->nprocs * sizeof(int));
        ::MPI_Allgather(&temp, 1, MPI_INT, pproc, 1, MPI_INT, world);

        sum = 0;
        for (int i = 0; i < comm->nprocs; ++i) { sum += pproc[i]; }
        --tries;

      } while ((sum > 0) && (tries > 0));
    }

    if (atom->natoms < natoms_previous) {
      post_delete();
    } else {
      post_add(nlocal_previous);
    }
  }

  if (comm->me == 0) {
    double newsupersaturation = compute_supersaturation_mono->compute_scalar();
    bigint atom_delta = std::abs(natoms_previous - atom->natoms);
    if (screenflag != 0) {
      utils::logmesg(lmp,
                     "fix SS: {} {} atoms. Previous SS: {:.3f}, new SS: {:.3f}, delta: {:.3f}. "
                     "Total atoms: {}",
                     delflag ? "deleted" : "added", atom_delta, previous_supersaturation,
                     newsupersaturation, newsupersaturation - previous_supersaturation,
                     atom->natoms);
    }
    if (fileflag != 0) {
      fmt::print(fp, "{},{},{},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f}\n", update->ntimestep, atom->natoms,
                 delflag ? delta : 0, !delflag ? delta : 0, delflag ? atom_delta : 0,
                 !delflag ? atom_delta : 0, previous_supersaturation, newsupersaturation,
                 newsupersaturation - previous_supersaturation, !delflag ? static_cast<double>(delta*100)/atom_delta : 0);
      ::fflush(fp);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixSupersaturation::delete_monomers() noexcept(true)
{
  // delete local atoms flagged in dlist
  // reset nlocal
  int nlocal = atom->nlocal;

  int local_monomers = compute_supersaturation_mono->local_monomers;
  const int *mono_idx = compute_supersaturation_mono->mono_idx;

  const int *mx = pproc[comm->me] > local_monomers ? &local_monomers : pproc + comm->me;

  while (*mx > 0) {
    atom->avec->copy(nlocal - 1, mono_idx[local_monomers - 1], 1);
    --pproc[comm->me];
    --local_monomers;
    --nlocal;
  }

  compute_supersaturation_mono->local_monomers = local_monomers;
  atom->nlocal = nlocal;
}

/* ---------------------------------------------------------------------- */

void FixSupersaturation::add_monomers_universe_random(bigint delta) noexcept(true)
{
  for (bigint i = 0; i < delta; ++i) {
    if (gen_one_universe()) {    // if success new coords will be already in xone[]
      if ((coord[0] >= subbonds[0][0]) && (coord[0] < subbonds[0][1]) &&
          (coord[1] >= subbonds[1][0]) && (coord[1] < subbonds[1][1]) &&
          (coord[2] >= subbonds[2][0]) && (coord[2] < subbonds[2][1])) {
        atom->avec->create_atom(ntype, xone);
        if (fix_temp != 0) { set_speed(atom->nlocal - 1); }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixSupersaturation::add_monomers_local_random() noexcept(true)
{
  int ninsert = 0;
  int unsucc = 0;
  for (int i = 0; i < pproc[comm->me]; ++i) {
    if (moveflag ? gen_one_local_move() : gen_one_local()) {
      unsucc = 0;
      atom->avec->create_atom(ntype, xone);

      if (fix_temp != 0) { set_speed(atom->nlocal - 1); }

      ++ninsert;
    } else {
      ++unsucc;
      if (unsucc > 10) { break; }
    }
  }
  pproc[comm->me] -= ninsert;
}

/* ---------------------------------------------------------------------- */

void FixSupersaturation::add_monomers_local_grid() noexcept(true)
{
  auto const nx =
      static_cast<bigint>(::floor((subbonds[0][1] - subbonds[0][0] - 2 * overlap) / overlap));
  auto const ny =
      static_cast<bigint>(::floor((subbonds[1][1] - subbonds[1][0] - 2 * overlap) / overlap));
  auto const nz =
      static_cast<bigint>(::floor((subbonds[2][1] - subbonds[2][0] - 2 * overlap) / overlap));
  for (bigint i = 0; (i < nx) && (pproc[comm->me] > 0); ++i) {
    for (bigint j = 0; (j < ny) && (pproc[comm->me] > 0); ++j) {
      for (bigint k = 0; (k < nz) && (pproc[comm->me] > 0); ++k) {
        if (moveflag ? gen_one_local_at_move(
                           subbonds[0][0] + overlap * (i + 1), subbonds[1][0] + overlap * (j + 1),
                           subbonds[2][0] + overlap * (k + 1), overlap, overlap, overlap)
                     : gen_one_local_at(
                           subbonds[0][0] + overlap * (i + 1), subbonds[1][0] + overlap * (j + 1),
                           subbonds[2][0] + overlap * (k + 1), overlap, overlap, overlap)) {
          atom->avec->create_atom(ntype, xone);
          if (fix_temp != 0) { set_speed(atom->nlocal - 1); }
          --pproc[comm->me];
        }
      }
    }
  }

  add_monomers_local_random();
}

/* ---------------------------------------------------------------------- */

void FixSupersaturation::set_speed(int pID) noexcept(true)
{
  double **v = atom->v;
  // generate velocities
  constexpr long double c_v = 0.7978845608028653558798921198687L;    // sqrt(2/pi)
  const double sigma = std::sqrt(monomer_temperature / atom->mass[ntype]);
  const double v_mean = static_cast<double>(c_v) * sigma;
  v[pID][0] = v_mean + vrandom->gaussian() * sigma;
  v[pID][1] = v_mean + vrandom->gaussian() * sigma;
  if (domain->dimension == 3) { v[pID][2] = v_mean + vrandom->gaussian() * sigma; }
}

/* ----------------------------------------------------------------------
  attempts to create coords up to maxtry times
  criteria for insertion: region, triclinic box, overlap
------------------------------------------------------------------------- */

bool FixSupersaturation::gen_one_universe() noexcept(true)
{

  int ntry = 0;
  bool success = false;

  while (ntry < maxtry) {
    ++ntry;

    // generate new random position
    xone[0] = globbonds[0][0] + xrandom->uniform() * (globbonds[0][1] - globbonds[0][0]);
    xone[1] = globbonds[1][0] + xrandom->uniform() * (globbonds[1][1] - globbonds[1][0]);
    xone[2] = globbonds[2][0] + xrandom->uniform() * (globbonds[2][1] - globbonds[2][0]);
    if (domain->dimension == 2) { xone[2] = 0.0; }

    if ((region != nullptr) && (region->match(xone[0], xone[1], xone[2]) == 0)) { continue; }

    if (triclinic != 0) {
      domain->x2lamda(xone, lamda);
      coord = lamda;
      if ((coord[0] < boxlo[0]) || (coord[0] >= boxhi[0]) || (coord[1] < boxlo[1]) ||
          (coord[1] >= boxhi[1]) || (coord[2] < boxlo[2]) || (coord[2] >= boxhi[2])) {
        continue;
      }
    } else {
      coord = xone;
    }

    // check for overlap of new atom/mol with all other atoms
    // minimum_image() needed to account for distances across PBC

    double **x = atom->x;
    int reject = 0;

    // check new position for overlapping with all local atoms
    for (int i = 0; i < atom->nmax; ++i) {
      double delx = xone[0] - x[i][0];
      double dely = xone[1] - x[i][1];
      double delz = xone[2] - x[i][2];

      const double distsq1 = delx * delx + dely * dely + delz * delz;
      domain->minimum_image(delx, dely, delz);
      const double distsq = delx * delx + dely * dely + delz * delz;
      if ((distsq < odistsq) || (distsq1 < odistsq)) {
        reject = 1;
        break;
      }
    }

    // gather reject flags from all of the procs
    int reject_any = 0;
    ::MPI_Allreduce(&reject, &reject_any, 1, MPI_INT, MPI_MAX, world);
    if (reject_any != 0) { continue; }

    // all tests passed

    success = true;
    break;
  }

  return success;

}    // bool FixSupersaturation::gen_one_full()

/* ---------------------------------------------------------------------- */

bool FixSupersaturation::gen_one_local() noexcept(true)
{
  int ntry = 0;

  while (ntry < maxtry) {
    ++ntry;

    // generate new random position
    xone[0] = subbonds[0][0] + xrandom->uniform() * (subbonds[0][1] - subbonds[0][0]);
    xone[1] = subbonds[1][0] + xrandom->uniform() * (subbonds[1][1] - subbonds[1][0]);
    xone[2] = subbonds[2][0] + xrandom->uniform() * (subbonds[2][1] - subbonds[2][0]);
    if (domain->dimension == 2) { xone[2] = 0.0; }

    if ((region != nullptr) && (region->match(xone[0], xone[1], xone[2]) == 0)) { continue; }

    if (triclinic != 0) {
      domain->x2lamda(xone, lamda);
      coord = lamda;
      if ((coord[0] < boxlo[0]) || (coord[0] >= boxhi[0]) || (coord[1] < boxlo[1]) ||
          (coord[1] >= boxhi[1]) || (coord[2] < boxlo[2]) || (coord[2] >= boxhi[2])) {
        continue;
      }
    } else {
      coord = xone;
    }

    // check for overlap of new atom/mol with all other atoms
    // minimum_image() needed to account for distances across PBC

    double **x = atom->x;
    bool reject = false;

    // check new position for overlapping with all local atoms
    for (int i = 0; i < atom->nmax; ++i) {
      double delx = xone[0] - x[i][0];
      double dely = xone[1] - x[i][1];
      double delz = xone[2] - x[i][2];
      const double distsq1 = delx * delx + dely * dely + delz * delz;

      domain->minimum_image(delx, dely, delz);
      const double distsq = delx * delx + dely * dely + delz * delz;
      if ((distsq < odistsq) || (distsq1 < odistsq)) {
        reject = true;
        break;
      }
    }

    if (reject) { continue; }

    return true;
  }

  return false;
}    // void FixKeepCount::gen_one_sub()

/* ---------------------------------------------------------------------- */

bool FixSupersaturation::gen_one_local_move() noexcept(true)
{
  int ntry = 0;

  while (ntry < maxtry) {
    ++ntry;

    // generate new random position
    xone[0] = subbonds[0][0] + xrandom->uniform() * (subbonds[0][1] - subbonds[0][0]);
    xone[1] = subbonds[1][0] + xrandom->uniform() * (subbonds[1][1] - subbonds[1][0]);
    xone[2] = subbonds[2][0] + xrandom->uniform() * (subbonds[2][1] - subbonds[2][0]);

    int ntry_move = 0;
    while (ntry_move < maxtry_move) {
      ++ntry_move;

      if (domain->dimension == 2) { xone[2] = 0.0; }

      if ((region != nullptr) && (region->match(xone[0], xone[1], xone[2]) == 0)) { continue; }

      if (triclinic != 0) {
        domain->x2lamda(xone, lamda);
        coord = lamda;
        if ((coord[0] < boxlo[0]) || (coord[0] >= boxhi[0]) || (coord[1] < boxlo[1]) ||
            (coord[1] >= boxhi[1]) || (coord[2] < boxlo[2]) || (coord[2] >= boxhi[2])) {
          continue;
        }
      } else {
        coord = xone;
      }

      // check for overlap of new atom/mol with all other atoms
      // minimum_image() needed to account for distances across PBC

      double **x = atom->x;
      bool reject = false;

      // check new position for overlapping with all local atoms
      for (int i = 0; i < atom->nmax; ++i) {
        double delx = xone[0] - x[i][0];
        double dely = xone[1] - x[i][1];
        double delz = xone[2] - x[i][2];
        const double distsq1 = delx * delx + dely * dely + delz * delz;

        domain->minimum_image(delx, dely, delz);
        const double distsq = delx * delx + dely * dely + delz * delz;
        if ((distsq < odistsq) || (distsq1 < odistsq)) {
          reject = true;

          xone[0] = x[i][0] + delx * ::sqrt(odistsq / distsq1);
          xone[1] = x[i][1] + dely * ::sqrt(odistsq / distsq1);
          xone[2] = x[i][2] + delz * ::sqrt(odistsq / distsq1);

          break;
        }
      }
      if (reject) { continue; }
    }

    return true;
  }

  return false;
}    // void FixKeepCount::gen_one_sub()

/* ---------------------------------------------------------------------- */

bool FixSupersaturation::gen_one_local_at(double _x, double _y, double _z, double _dx, double _dy,
                                          double _dz) noexcept(true)
{
  int ntry = 0;

  while (ntry < 9) {
    ++ntry;

    // generate new random position
    xone[0] = _x + xrandom->uniform() * _dx;
    xone[1] = _y + xrandom->uniform() * _dy;
    xone[2] = _z + xrandom->uniform() * _dz;
    if (domain->dimension == 2) { xone[2] = 0.0; }

    if ((region != nullptr) && (region->match(xone[0], xone[1], xone[2]) == 0)) { continue; }

    if (triclinic != 0) {
      domain->x2lamda(xone, lamda);
      coord = lamda;
      if ((coord[0] < boxlo[0]) || (coord[0] >= boxhi[0]) || (coord[1] < boxlo[1]) ||
          (coord[1] >= boxhi[1]) || (coord[2] < boxlo[2]) || (coord[2] >= boxhi[2])) {
        continue;
      }
    } else {
      coord = xone;
    }

    // check for overlap of new atom/mol with all other atoms
    // minimum_image() needed to account for distances across PBC

    double **x = atom->x;
    bool reject = false;

    // check new position for overlapping with all local atoms
    for (int i = 0; i < atom->nmax; ++i) {
      double delx = xone[0] - x[i][0];
      double dely = xone[1] - x[i][1];
      double delz = xone[2] - x[i][2];
      double const distsq1 = delx * delx + dely * dely + delz * delz;

      domain->minimum_image(delx, dely, delz);
      double const distsq = delx * delx + dely * dely + delz * delz;
      if ((distsq < odistsq) || (distsq1 < odistsq)) {
        reject = true;
        break;
      }
    }

    if (reject) { continue; }

    return true;
  }

  return false;
}    // void FixKeepCount::gen_one()

/* ---------------------------------------------------------------------- */

bool FixSupersaturation::gen_one_local_at_move(double _x, double _y, double _z, double _dx,
                                               double _dy, double _dz) noexcept(true)
{
  int ntry = 0;

  // generate new random position
  xone[0] = _x + _dx / 2;
  xone[1] = _y + _dy / 2;
  xone[2] = _z + _dz / 2;

  while (ntry < maxtry_move) {
    ++ntry;
    if (domain->dimension == 2) { xone[2] = 0.0; }

    if ((region != nullptr) && (region->match(xone[0], xone[1], xone[2]) == 0)) { continue; }

    if (triclinic != 0) {
      domain->x2lamda(xone, lamda);
      coord = lamda;
      if ((coord[0] < boxlo[0]) || (coord[0] >= boxhi[0]) || (coord[1] < boxlo[1]) ||
          (coord[1] >= boxhi[1]) || (coord[2] < boxlo[2]) || (coord[2] >= boxhi[2])) {
        continue;
      }
    } else {
      coord = xone;
    }

    // check for overlap of new atom/mol with all other atoms
    // minimum_image() needed to account for distances across PBC

    double **x = atom->x;
    bool reject = false;

    // check new position for overlapping with all local atoms
    for (int i = 0; i < atom->nmax; ++i) {
      double delx = xone[0] - x[i][0];
      double dely = xone[1] - x[i][1];
      double delz = xone[2] - x[i][2];
      double const distsq1 = delx * delx + dely * dely + delz * delz;

      domain->minimum_image(delx, dely, delz);
      double const distsq = delx * delx + dely * dely + delz * delz;
      if ((distsq < odistsq) || (distsq1 < odistsq)) {
        reject = true;

        xone[0] = x[i][0] + delx * ::sqrt(odistsq / distsq1);
        xone[1] = x[i][1] + dely * ::sqrt(odistsq / distsq1);
        xone[2] = x[i][2] + delz * ::sqrt(odistsq / distsq1);

        break;
      }
    }

    if (reject) { continue; }

    return true;
  }

  return false;
}    // void FixKeepCount::gen_one()

/* ---------------------------------------------------------------------- */

void FixSupersaturation::post_add(const int nlocal_previous) noexcept(true)
{
  // init per-atom fix/compute/variable values for created atoms

  atom->data_fix_compute_variable(nlocal_previous, atom->nlocal);

  // set new total # of atoms and error check

  bigint nblocal = atom->nlocal;
  ::MPI_Allreduce(&nblocal, &atom->natoms, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  if ((atom->natoms < 0) || (atom->natoms >= MAXBIGINT)) {
    error->all(FLERR, "Too many total atoms");
  }

  // add IDs for newly created atoms
  // check that atom IDs are valid

  if (atom->tag_enable != 0) { atom->tag_extend(); }
  atom->tag_check();

  // if global map exists, reset it
  // invoke map_init() b/c atom count has grown

  if (atom->map_style != Atom::MAP_NONE) {
    atom->map_init();
    atom->map_set();
  }
}

/* ---------------------------------------------------------------------- */

void FixSupersaturation::post_delete() noexcept(true)
{
  if (atom->molecular == Atom::ATOMIC) {
    tagint *tag = atom->tag;
    int const nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; ++i) { tag[i] = 0; }
    atom->tag_extend();
  }

  // reset atom->natoms and also topology counts

  bigint nblocal = atom->nlocal;
  ::MPI_Allreduce(&nblocal, &atom->natoms, 1, MPI_LMP_BIGINT, MPI_SUM, world);

  // reset bonus data counts

  const auto *avec_ellipsoid = dynamic_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
  const auto *avec_line = dynamic_cast<AtomVecLine *>(atom->style_match("line"));
  const auto *avec_tri = dynamic_cast<AtomVecTri *>(atom->style_match("tri"));
  const auto *avec_body = dynamic_cast<AtomVecBody *>(atom->style_match("body"));
  bigint nlocal_bonus = 0;

  if (atom->nellipsoids > 0) {
    nlocal_bonus = avec_ellipsoid->nlocal_bonus;
    ::MPI_Allreduce(&nlocal_bonus, &atom->nellipsoids, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  }
  if (atom->nlines > 0) {
    nlocal_bonus = avec_line->nlocal_bonus;
    ::MPI_Allreduce(&nlocal_bonus, &atom->nlines, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  }
  if (atom->ntris > 0) {
    nlocal_bonus = avec_tri->nlocal_bonus;
    ::MPI_Allreduce(&nlocal_bonus, &atom->ntris, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  }
  if (atom->nbodies > 0) {
    nlocal_bonus = avec_body->nlocal_bonus;
    ::MPI_Allreduce(&nlocal_bonus, &atom->nbodies, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  }

  // reset atom->map if it exists
  // set nghost to 0 so old ghosts of deleted atoms won't be mapped

  if (atom->map_style != Atom::MAP_NONE) {
    atom->nghost = 0;
    atom->map_init();
    atom->map_set();
  }
}

/* ---------------------------------------------------------------------- */
