/*
 ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "fix_supersaturation.h"
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
#include <ctime>

using namespace LAMMPS_NS;
using namespace FixConst;

constexpr int DEFAULT_MAXTRY = 1000;
constexpr int DEFAULT_MAXTRY_CALL = 5;

/* ---------------------------------------------------------------------- */

FixSupersaturation::FixSupersaturation(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), screenflag(1), fileflag(0), next_step(0), maxtry(DEFAULT_MAXTRY),
    scaleflag(0), fix_temp(0), maxtry_call(DEFAULT_MAXTRY_CALL), offflag(0)
{

  restart_pbc = 1;
  nevery = 1;

  if (narg < 10) { utils::missing_cmd_args(FLERR, "cluster/crush", error); }

  // Parse arguments //

  // Target region
  region = domain->get_region_by_id(arg[3]);
  if (region == nullptr) { error->all(FLERR, "Cannot find target region {}", arg[3]); }

  // Get compute supersaturation/mono
  compute_supersaturation_mono =
      static_cast<ComputeSupersaturationMono *>(modify->get_compute_by_id(arg[4]));
  if (compute_supersaturation_mono == nullptr) {
    error->all(FLERR,
               "fix supersaturation: cannot find compute of style 'supersaturation/mono' with "
               "given id: {}",
               arg[4]);
  }

  // Minimum distance to other atoms from the place atom teleports to
  double overlap = utils::numeric(FLERR, arg[5], true, lmp);
  if (overlap < 0) {
    error->all(FLERR, "Minimum distance for fix cluster/crush must be non-negative");
  }
  // apply scaling factor for styles that use distance-dependent factors
  overlap *= domain->lattice->xlattice;
  odistsq = overlap * overlap;

  // # of type of atoms to insert
  ntype = utils::inumeric(FLERR, arg[6], true, lmp);

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
  if (damp <= 0 || damp > 1) {
    error->all(FLERR, "Dampfing parameter for fix supersaturation must be in range (0,1]");
  }

  // Parse optional keywords

  int iarg = 10;
  fp = nullptr;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "maxtry") == 0) {

      // Max attempts to search for a new suitable location
      maxtry = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      if (maxtry < 1) { error->all(FLERR, "maxtry for cluster/crush cannot be less than 1"); }
      iarg += 2;

    } else if (strcmp(arg[iarg], "maxtry_call") == 0) {

      // Get max number of tries for calling delete_monomers()/add_monomers()
      maxtry_call = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      if (maxtry_call < 1) {
        error->all(FLERR, "maxtry_call for cluster/crush cannot be less than 1");
      }
      iarg += 2;

    } else if (strcmp(arg[iarg], "temp") == 0) {

      // Monomer temperature
      fix_temp = 1;
      monomer_temperature = utils::numeric(FLERR, arg[iarg + 1], true, lmp);
      if (monomer_temperature < 0) {
        error->all(FLERR, "Monomer temperature for cluster/crush cannot be negative");
      }

      // Get the seed for velocity generator
      int const vseed = utils::inumeric(FLERR, arg[iarg + 2], true, lmp);
      vrandom = new RanPark(lmp, vseed);
      iarg += 3;

    } else if (strcmp(arg[iarg], "noscreen") == 0) {

      // Do not output to screen
      screenflag = 0;
      iarg += 1;

    } else if (strcmp(arg[iarg], "file") == 0) {

      // Write output to new file
      if (comm->me == 0) {
        fileflag = 1;
        fp = fopen(arg[iarg + 1], "w");
        if (fp == nullptr) {
          error->one(FLERR, "Cannot open cluster/crush stats file {}: {}", arg[iarg + 1],
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
          error->one(FLERR, "Cannot open cluster/crush stats file {}: {}", arg[iarg + 1],
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
        error->all(FLERR, "start_offset for cluster/crush cannot be less than 0");
      }
      offflag = 1;
      iarg += 2;

    } else if (strcmp(arg[iarg], "units") == 0) {

      if (strcmp(arg[iarg + 1], "box") == 0) {
        scaleflag = 0;
      } else if (strcmp(arg[iarg + 1], "lattice") == 0) {
        scaleflag = 1;
      } else {
        error->all(FLERR, "Unknown cluster/crush units option {}", arg[iarg + 1]);
      }
      iarg += 2;

    } else {
      error->all(FLERR, "Illegal cluster/crush command option {}", arg[iarg]);
    }
  }

  triclinic = domain->triclinic;

  // bounding box for atom creation
  // only limit bbox by region if its bboxflag is set (interior region)

  if (triclinic == 0) {
    xlo = domain->sublo[0];
    xhi = domain->subhi[0];
    ylo = domain->sublo[1];
    yhi = domain->subhi[1];
    zlo = domain->sublo[2];
    zhi = domain->subhi[2];
  } else {
    xlo = domain->sublo_lamda[0] * domain->prd[0];
    xhi = domain->subhi_lamda[0] * domain->prd[0];
    ylo = domain->sublo_lamda[1] * domain->prd[1];
    yhi = domain->subhi_lamda[1] * domain->prd[1];
    zlo = domain->sublo_lamda[2] * domain->prd[2];
    zhi = domain->subhi_lamda[2] * domain->prd[2];
    boxlo = domain->boxlo_lamda;
    boxhi = domain->boxhi_lamda;
  }

  if ((region != nullptr) && (region->bboxflag != 0)) {
    xlo = MAX(xlo, region->extent_xlo);
    xhi = MIN(xhi, region->extent_xhi);
    ylo = MAX(ylo, region->extent_ylo);
    yhi = MIN(yhi, region->extent_yhi);
    zlo = MAX(zlo, region->extent_zlo);
    zhi = MIN(zhi, region->extent_zhi);
  }

  if (xlo > xhi || ylo > yhi || zlo > zhi) {
    error->all(FLERR, "No overlap of box and region for fix supersaturation");
  }

  if (comm->me == 0 && (fileflag != 0)) {
    fmt::print(fp, "ntimestep,a2d,a2a,ad,aa,ssb,ssa,del\n");
    fflush(fp);
  }

  next_step = update->ntimestep - (update->ntimestep % nevery);
  if (offflag) {
    next_step = update->ntimestep + start_offset;
  }

  memory->create(pproc, comm->nprocs * sizeof(int), "fix_supersaturation:pproc");

  memset(xone, 0, 3 * sizeof(double));
  memset(lamda, 0, 3 * sizeof(double));

  srand(time(nullptr));

  if (comm->me == 0) { log = fopen("fix_super.log", "a"); }
}

/* ---------------------------------------------------------------------- */

FixSupersaturation::~FixSupersaturation()
{
  delete xrandom;
  delete vrandom;
  if ((fp != nullptr) && (comm->me == 0)) {
    fflush(fp);
    fclose(fp);
  }
  memory->destroy(pproc);
  if (comm->me == 0) {
    fflush(log);
    fclose(log);
  }
}

/* ---------------------------------------------------------------------- */

int FixSupersaturation::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  // mask |= POST_NEIGHBOR;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSupersaturation::init() {}

/* ---------------------------------------------------------------------- */

void FixSupersaturation::pre_exchange()
{
  if (update->ntimestep < next_step) {
    if (comm->me == 0) {
      fmt::print(log, "{}, returned\n", update->ntimestep);
      fflush(log);
    }
    return;
  }
  next_step = update->ntimestep + nevery;
  if (comm->me == 0) {
    fmt::print(log, "{}, not returned\n", update->ntimestep);
    fflush(log);
  }

  if (compute_supersaturation_mono->invoked_scalar != update->ntimestep) {
    compute_supersaturation_mono->compute_scalar();
  }
  double previous_supersaturation = compute_supersaturation_mono->scalar;
  if (comm->me == 0) {
    fmt::print(log, "Previous supersaturation: {}\n", previous_supersaturation);
    fflush(log);
  }

  auto delta = static_cast<bigint>(
      round(compute_supersaturation_mono->execute_func() * domain->volume() * supersaturation -
            compute_supersaturation_mono->global_monomers));
  if (comm->me == 0) {
    fmt::print(log, "delta: {}\n", delta);
    fflush(log);
  }
  const bool delflag = delta < 0;
  delta = static_cast<bigint>(round(damp * std::abs(delta)));
  if (comm->me == 0) {
    fmt::print(log, "delflag: {}, damp*abs(delta)={}\n", delflag ? "true" : "false", delta);
    fflush(log);
  }

  if (delta != 0) {
    bigint const natoms_previous = atom->natoms;
    int const nlocal_previous = atom->nlocal;

    bigint sum = delta;
    memset(pproc, 0, comm->nprocs * sizeof(int));
    int __ntry = maxtry_call;
    if (comm->me == 0) {
      fmt::print(log, "Starting loop\n");
      fflush(log);
    }

    do {
      if (comm->me == 0) {
        fmt::print(log, "\n");
        fflush(log);
      }

      int additional_proc = rand() % comm->nprocs;
      int for_every = sum / comm->nprocs;
      pproc[comm->me] = comm->me == additional_proc ? for_every + sum % comm->nprocs : for_every;

      if (comm->me == 0) {
        fmt::print(log, "Sum: {}\n", sum);
        fmt::print(log, "__ntry: {}\n", __ntry);
        fmt::print(log, "Additional proc: {}\n", additional_proc);
        for (int i = 0; i < comm->nprocs - 1; ++i) {
          if (i == additional_proc) {
            fmt::print(log, "{} ", for_every + sum % comm->nprocs);
          } else {
            fmt::print(log, "{} ", for_every);
          }
        }
        fmt::print(log, "\n", for_every);

        fflush(log);
      }

      if (pproc[comm->me] > 0) {
        if (delflag) {
          if (comm->me == 0) {
            fmt::print(log, "Deleting\n");
            fflush(log);
          }
          delete_monomers();
        } else {
          if (comm->me == 0) {
            fmt::print(log, "Adding\n");
            fflush(log);
          }
          add_monomers2();
        }
      }

      if (comm->me == 0) {
        fmt::print(log, "After modification\n");
        fflush(log);
      }

      int temp = pproc[comm->me];
      memset(pproc, 0, comm->nprocs * sizeof(int));
      MPI_Allgather(&temp, 1, MPI_INT, pproc, 1, MPI_INT, world);

      if (comm->me == 0) {
        for (int i = 0; i < comm->nprocs - 1; ++i) { fmt::print(log, "{} ", pproc[i]); }
        fmt::print(log, "\n", for_every);

        fflush(log);
      }

      sum = 0;
      for (int i = 0; i < comm->nprocs; ++i) { sum += pproc[i]; }
      --__ntry;

      if (comm->me == 0) {
        fmt::print(log, "Sum: {}\n", sum);
        fmt::print(log, "\n");
        fflush(log);
      }
    } while (sum > 0 && __ntry > 0);

    if (comm->me == 0) {
      fmt::print(log, "after loop\n");
      fflush(log);
    }

    if (delflag) {
      if (atom->molecular == Atom::ATOMIC) {
        tagint *tag = atom->tag;
        int const nlocal = atom->nlocal;
        for (int i = 0; i < nlocal; i++) { tag[i] = 0; }
        atom->tag_extend();
      }

      // reset atom->natoms and also topology counts

      bigint nblocal = atom->nlocal;
      MPI_Allreduce(&nblocal, &atom->natoms, 1, MPI_LMP_BIGINT, MPI_SUM, world);

      // reset bonus data counts

      auto *avec_ellipsoid = static_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
      auto *avec_line = static_cast<AtomVecLine *>(atom->style_match("line"));
      auto *avec_tri = static_cast<AtomVecTri *>(atom->style_match("tri"));
      auto *avec_body = static_cast<AtomVecBody *>(atom->style_match("body"));
      bigint nlocal_bonus = 0;

      if (atom->nellipsoids > 0) {
        nlocal_bonus = avec_ellipsoid->nlocal_bonus;
        MPI_Allreduce(&nlocal_bonus, &atom->nellipsoids, 1, MPI_LMP_BIGINT, MPI_SUM, world);
      }
      if (atom->nlines > 0) {
        nlocal_bonus = avec_line->nlocal_bonus;
        MPI_Allreduce(&nlocal_bonus, &atom->nlines, 1, MPI_LMP_BIGINT, MPI_SUM, world);
      }
      if (atom->ntris > 0) {
        nlocal_bonus = avec_tri->nlocal_bonus;
        MPI_Allreduce(&nlocal_bonus, &atom->ntris, 1, MPI_LMP_BIGINT, MPI_SUM, world);
      }
      if (atom->nbodies > 0) {
        nlocal_bonus = avec_body->nlocal_bonus;
        MPI_Allreduce(&nlocal_bonus, &atom->nbodies, 1, MPI_LMP_BIGINT, MPI_SUM, world);
      }

      // reset atom->map if it exists
      // set nghost to 0 so old ghosts of deleted atoms won't be mapped

      if (atom->map_style != Atom::MAP_NONE) {
        atom->nghost = 0;
        atom->map_init();
        atom->map_set();
      }

    } else {
      // init per-atom fix/compute/variable values for created atoms

      atom->data_fix_compute_variable(nlocal_previous, atom->nlocal);

      // set new total # of atoms and error check

      bigint nblocal = atom->nlocal;
      MPI_Allreduce(&nblocal, &atom->natoms, 1, MPI_LMP_BIGINT, MPI_SUM, world);
      if (atom->natoms < 0 || atom->natoms >= MAXBIGINT) {
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

    double newsupersaturation = compute_supersaturation_mono->compute_scalar();
    if (comm->me == 0) {
      bigint atom_delta = std::abs(natoms_previous - atom->natoms);
      if (screenflag != 0) {
        fmt::print(fp, "fix SS: {} {} atoms. Previous SS: {:.3f}, new SS: {:.3f}, delta: {:.3f}",
                   delflag ? "deleted" : "added", atom_delta, previous_supersaturation,
                   newsupersaturation, newsupersaturation - previous_supersaturation);
      }
      if (fileflag != 0) {
        fmt::print(fp, "{},{},{},{},{},{:.3f},{:.3f},{:.3f}\n", update->ntimestep,
                   delflag ? delta : 0, !delflag ? delta : 0, delflag ? atom_delta : 0,
                   !delflag ? atom_delta : 0, previous_supersaturation, newsupersaturation,
                   newsupersaturation - previous_supersaturation);
        fflush(fp);
      }
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
  int *mono_idx = compute_supersaturation_mono->mono_idx;

  int *mx = pproc[comm->me] > local_monomers ? &local_monomers : pproc + comm->me;

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

void FixSupersaturation::add_monomers() noexcept(true)
{
  int ninsert = 0;
  int unsucc = 0;
  for (int i = 0; i < pproc[comm->me]; ++i) {
    if (comm->me == 0) {
      fmt::print(log, "Generating...");
      fflush(log);
    }
    if (gen_one()) {
      unsucc = 0;
      if (comm->me == 0) {
        fmt::print(log, "Creating...");
        fflush(log);
      }
      atom->avec->create_atom(ntype, xone);
      if (comm->me == 0) {
        fmt::print(log, "Created\n");
        fflush(log);
      }

      if (fix_temp != 0) { set_speed(); }

      ++ninsert;
    } else {
      ++unsucc;
      if (comm->me == 0) {
        fmt::print(log, "Unsuccesful\n");
        fflush(log);
      }
      if (unsucc > 10) {
        if (comm->me == 0) {
          fmt::print(log, "Skipping...\n");
          fflush(log);
        }
        break;
      }
    }
  }
  pproc[comm->me] -= ninsert;
}

void FixSupersaturation::add_monomers2() noexcept(true)
{
  int ninsert = 0;
  int unsucc = 0;
  for (int i = 0; i < pproc[comm->me]; ++i) {
    if (gen_one(xlo + odistsq * i, ylo + odistsq * i, zlo + odistsq * i, odistsq, odistsq,
                odistsq)) {
      unsucc = 0;
      atom->avec->create_atom(ntype, xone);
      if (fix_temp != 0) { set_speed(); }
      ++ninsert;
    } else {
      ++unsucc;
      if (unsucc > 10) { break; }
    }
  }
  pproc[comm->me] -= ninsert;
  add_monomers();
}

void FixSupersaturation::set_speed() noexcept(true)
{
  double **v = atom->v;
  const int pID = atom->nlocal - 1;
  // generate velocities
  constexpr long double c_v = 0.7978845608028653558798921198687L;    // sqrt(2/pi)
  const double sigma = std::sqrt(monomer_temperature / atom->mass[ntype]);
  const double v_mean = c_v * sigma;
  v[pID][0] = v_mean + vrandom->gaussian() * sigma;
  v[pID][1] = v_mean + vrandom->gaussian() * sigma;
  if (domain->dimension == 3) { v[pID][2] = v_mean + vrandom->gaussian() * sigma; }
}

/* ----------------------------------------------------------------------
  attempts to create coords up to maxtry times
  criteria for insertion: region, triclinic box, overlap
------------------------------------------------------------------------- */

bool FixSupersaturation::gen_one() noexcept(true)
{

  int ntry = 0;
  bool success = false;

  while (ntry < maxtry) {
    ++ntry;

    // generate new random position
    xone[0] = xlo + xrandom->uniform() * (xhi - xlo);
    xone[1] = ylo + xrandom->uniform() * (yhi - ylo);
    xone[2] = zlo + xrandom->uniform() * (zhi - zlo);

    if ((region != nullptr) && (region->match(xone[0], xone[1], xone[2]) == 0)) { continue; }

    if (triclinic != 0) {
      domain->x2lamda(xone, lamda);
      if (lamda[0] < boxlo[0] || lamda[0] >= boxhi[0] || lamda[1] < boxlo[1] ||
          lamda[1] >= boxhi[1] || lamda[2] < boxlo[2] || lamda[2] >= boxhi[2]) {
        continue;
      }
    }

    // check for overlap of new atom/mol with all other atoms
    // minimum_image() needed to account for distances across PBC

    double **x = atom->x;
    int reject = 0;

    // check new position for overlapping with all local atoms
    for (int i = 0; i < atom->nmax; i++) {
      double delx, dely, delz, distsq, distsq1;

      delx = xone[0] - x[i][0];
      dely = xone[1] - x[i][1];
      delz = xone[2] - x[i][2];
      distsq1 = delx * delx + dely * dely + delz * delz;
      domain->minimum_image(delx, dely, delz);
      distsq = delx * delx + dely * dely + delz * delz;
      if (distsq < odistsq || distsq1 < odistsq) {
        reject = 1;
        break;
      }
    }

    if (reject != 0) { continue; }

    // all tests passed

    success = true;
    break;
  }

  return success;

}    // void FixSupersaturation::gen_one()

bool FixSupersaturation::gen_one(double _x, double _y, double _z, double _dx, double _dy,
                                 double _dz) noexcept(true)
{

  int ntry = 0;
  bool success = false;

  while (ntry < 9) {
    ++ntry;

    // generate new random position
    xone[0] = _x + xrandom->uniform() * _dx;
    xone[1] = _y + xrandom->uniform() * _dy;
    xone[2] = _x + xrandom->uniform() * _dz;

    if ((region != nullptr) && (region->match(xone[0], xone[1], xone[2]) == 0)) { continue; }

    if (triclinic != 0) {
      domain->x2lamda(xone, lamda);
      if (lamda[0] < boxlo[0] || lamda[0] >= boxhi[0] || lamda[1] < boxlo[1] ||
          lamda[1] >= boxhi[1] || lamda[2] < boxlo[2] || lamda[2] >= boxhi[2]) {
        continue;
      }
    }

    // check for overlap of new atom/mol with all other atoms
    // minimum_image() needed to account for distances across PBC

    double **x = atom->x;
    int reject = 0;

    // check new position for overlapping with all local atoms
    for (int i = 0; i < atom->nmax; i++) {
      double delx, dely, delz, distsq, distsq1;

      delx = xone[0] - x[i][0];
      dely = xone[1] - x[i][1];
      delz = xone[2] - x[i][2];
      distsq1 = delx * delx + dely * dely + delz * delz;
      domain->minimum_image(delx, dely, delz);
      distsq = delx * delx + dely * dely + delz * delz;
      if (distsq < odistsq || distsq1 < odistsq) {
        reject = 1;
        break;
      }
    }

    if (reject != 0) { continue; }

    // all tests passed

    success = true;
    break;
  }

  return success;

}    // void FixSupersaturation::gen_one()

/* ---------------------------------------------------------------------- */