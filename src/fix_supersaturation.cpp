/*
 ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "fix_supersaturation.h"
#include <compute_supersaturation_mono.h>

#include "atom.h"
#include "atom_vec.h"
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
#include <unordered_set>

using namespace LAMMPS_NS;
using namespace FixConst;

constexpr int DEFAULT_MAXTRY = 1000;
constexpr int DEFAULT_MAXTRY_CALL = 5;
// constexpr double DEFAULT_DELTA = 0.01;


/* ---------------------------------------------------------------------- */

FixSupersaturation::FixSupersaturation(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{

  restart_pbc = 1;
  nevery = 1;

  fix_temp = 0;
  scaleflag = 0;
  screenflag = 1;
  fileflag = 0;
  next_step = 0;
  maxtry = DEFAULT_MAXTRY;
  maxtry_call = DEFAULT_MAXTRY_CALL;

  if (narg < 8) utils::missing_cmd_args(FLERR, "cluster/crush", error);

  // Parse arguments //

  // Target region
  region = domain->get_region_by_id(arg[3]);
  if (region == nullptr) { error->all(FLERR, "Cannot find target region {}", arg[3]); }

  // Get compute supersaturation/mono
  compute_supersaturation_mono = static_cast<ComputeSupersaturationMono*>(modify->get_compute_by_id(arg[4]));
  if (compute_supersaturation_mono == nullptr) error->all(FLERR, "fix supersaturation: cannot find compute of style 'supersaturation/mono' with given id: {}", arg[4]);

  // Minimum distance to other atoms from the place atom teleports to
  double overlap = utils::numeric(FLERR, arg[5], true, lmp);
  if (overlap < 0) error->all(FLERR, "Minimum distance for fix cluster/crush must be non-negative");
  // apply scaling factor for styles that use distance-dependent factors
  overlap *= domain->lattice->xlattice;
  odistsq = overlap * overlap;

  // # of type of atoms to insert
  int ntype = utils::inumeric(FLERR, arg[6], true, lmp);

  // Get the seed for coordinate generator
  int xseed = utils::numeric(FLERR, arg[7], true, lmp);
  xrandom = new RanPark(lmp, xseed);

  // Parse optional keywords

  int iarg = 8;
  fp = nullptr;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "maxtry") == 0) {

      // Max attempts to search for a new suitable location
      maxtry = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      if (maxtry < 1) error->all(FLERR, "maxtry for cluster/crush cannot be less than 1");
      iarg += 2;

    } else if (strcmp(arg[iarg], "maxtry_call") == 0) {

      // Get max number of tries for calling delete_monomers()/add_monomers()
      maxtry_call = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      if (maxtry_call < 1) error->all(FLERR, "maxtry_call for cluster/crush cannot be less than 1");
      iarg += 2;

    } else if (strcmp(arg[iarg], "temp") == 0) {

      // Monomer temperature
      fix_temp = 1;
      monomer_temperature = utils::numeric(FLERR, arg[iarg + 1], true, lmp);
      if (monomer_temperature < 0)
        error->all(FLERR, "Monomer temperature for cluster/crush cannot be negative");

      // Get the seed for velocity generator
      int vseed = utils::numeric(FLERR, arg[iarg + 2], true, lmp);
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
        if (fp == nullptr)
          error->one(FLERR, "Cannot open cluster/crush stats file {}: {}", arg[iarg + 1],
                     utils::getsyserror());
      }
      iarg += 2;

    } else if (strcmp(arg[iarg], "append") == 0) {

      // Append output to file
      if (comm->me == 0) {
        fileflag = 1;
        fp = fopen(arg[iarg + 1], "a");
        if (fp == nullptr)
          error->one(FLERR, "Cannot open cluster/crush stats file {}: {}", arg[iarg + 1],
                     utils::getsyserror());
      }
      iarg += 2;

    } else if (strcmp(arg[iarg], "nevery") == 0) {

      // Get execution period
      nevery = utils::numeric(FLERR, arg[iarg + 1], true, lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg], "units") == 0) {

      if (strcmp(arg[iarg + 1], "box") == 0)
        scaleflag = 0;
      else if (strcmp(arg[iarg + 1], "lattice") == 0)
        scaleflag = 1;
      else
        error->all(FLERR, "Unknown cluster/crush units option {}", arg[iarg + 1]);
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

  if (region && region->bboxflag) {
    xlo = MAX(xlo, region->extent_xlo);
    xhi = MIN(xhi, region->extent_xhi);
    ylo = MAX(ylo, region->extent_ylo);
    yhi = MIN(yhi, region->extent_yhi);
    zlo = MAX(zlo, region->extent_zlo);
    zhi = MIN(zhi, region->extent_zhi);
  }

  if (xlo > xhi || ylo > yhi || zlo > zhi)
    error->all(FLERR, "No overlap of box and region for fix supersaturation");

  if (comm->me == 0 && fileflag) {
    fmt::print(fp, "ntimestep,a2d,a2a,ad,aa,ssb,ssa,del\n");
    fflush(fp);
  }

  next_step = update->ntimestep - (update->ntimestep % nevery);

  memory->create(pproc, comm->nprocs * sizeof(int), "fix_supersaturation:pproc");

  memset(xone, 0, 3 * sizeof(double));
  memset(lamda, 0, 3 * sizeof(double));

  srand(time(NULL));
}

/* ---------------------------------------------------------------------- */

FixSupersaturation::~FixSupersaturation()
{
  delete xrandom;
  if (vrandom) delete vrandom;
  if (fp && (comm->me == 0)) fclose(fp);
  memory->destroy(pproc);
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
  if (update->ntimestep < next_step) return;
  next_step = update->ntimestep + nevery;

  if (compute_supersaturation_mono->invoked_scalar != update->ntimestep) {compute_supersaturation_mono->compute_scalar(); }
  double previous_supersaturation = compute_supersaturation_mono->scalar;

  bigint delta = static_cast<bigint>(round(compute_supersaturation_mono->execute_func() * domain->volume() * supersaturation - compute_supersaturation_mono->global_monomers));
  int delflag = delta > 0;
  delta = abs(delta);

  if (delta != 0){
    bigint natoms_previous = atom->natoms;
    int nlocal_previous = atom->nlocal;

    bigint sum = delta;
    memset(pproc, 0, comm->nprocs * sizeof(int));
    int __ntry = maxtry_call;

    do {
      pproc[comm->me] = comm->me == rand() % comm->nprocs ? sum / comm->nprocs + sum % comm->nprocs : sum / comm->nprocs;

      if (pproc[comm->me] > 0) {
        if (delflag){
          delete_monomers();
        } else {
          add_monomers();
        }
      }

      int temp = pproc[comm->me];
      memset(pproc, 0, comm->nprocs * sizeof(int));
      MPI_Allgather(&temp, 1, MPI_INT, pproc, 1, MPI_INT, world);

      for (int i = 0; i < comm->nprocs; ++i) sum += pproc[i];
      --__ntry;
    } while (sum > 0 && __ntry > 0);

    if (delflag){
      if (atom->molecular == Atom::ATOMIC) {
        tagint *tag = atom->tag;
        int nlocal = atom->nlocal;
        for (int i = 0; i < nlocal; i++) tag[i] = 0;
        atom->tag_extend();
      }
    } else {
      // init per-atom fix/compute/variable values for created atoms

      atom->data_fix_compute_variable(nlocal_previous, atom->nlocal);

      // set new total # of atoms and error check

      bigint nblocal = atom->nlocal;
      MPI_Allreduce(&nblocal, &atom->natoms, 1, MPI_LMP_BIGINT, MPI_SUM, world);
      if (atom->natoms < 0 || atom->natoms >= MAXBIGINT) error->all(FLERR, "Too many total atoms");

      // add IDs for newly created atoms
      // check that atom IDs are valid

      if (atom->tag_enable) atom->tag_extend();
      atom->tag_check();

      // if global map exists, reset it
      // invoke map_init() b/c atom count has grown

      if (atom->map_style != Atom::MAP_NONE) {
        atom->map_init();
        atom->map_set();
      }
    }

    double newsupersaturation = compute_supersaturation_mono->compute_scalar();
    if (comm->me == 0){
      bigint atom_delta = abs(natoms_previous - atom->natoms);
      if (screenflag) fmt::print(fp, "fix SS: {} {} atoms. Previous SS: {:.3f}, new SS: {:.3f}, delta: {:.3f}",
        delflag ? "deleted" : "added",
        atom_delta,
        previous_supersaturation,
        newsupersaturation,
        newsupersaturation - previous_supersaturation);
      if (fileflag) {
        fmt::print(fp, "{},{},{},{},{},{:.3f},{:.3f},{:.3f}\n",
            update->ntimestep,
             delflag ? delta : 0,
            !delflag ? delta : 0,
             delflag ? atom_delta : 0,
            !delflag ? atom_delta : 0,
            previous_supersaturation,
            newsupersaturation,
            newsupersaturation - previous_supersaturation
          );
        fflush(fp);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixSupersaturation::delete_monomers() noexcept(true) {
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

void FixSupersaturation::add_monomers() noexcept(true) {
  int ninsert = 0;
  for (int i = 0; i < pproc[comm->me]; ++i){
    if (gen_one()) {
      atom->avec->create_atom(ntype, xone);

      if (fix_temp) {
        double **v = atom->v;
        int pID = atom->nlocal - 1;
        // generate velocities
        constexpr long double c_v = 0.7978845608028653558798921198687L;    // sqrt(2/pi)
        double sigma = std::sqrt(monomer_temperature / atom->mass[ntype]);
        double v_mean = c_v * sigma;
        v[pID][0] = v_mean + vrandom->gaussian() * sigma;
        v[pID][1] = v_mean + vrandom->gaussian() * sigma;
        if (domain->dimension == 3) { v[pID][2] = v_mean + vrandom->gaussian() * sigma; }
      }

      ++ninsert;
    }
  }
  pproc[comm->me] -= ninsert;
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

    if (region && (region->match(xone[0], xone[1], xone[2]) == 0)) continue;

    if (triclinic) {
      domain->x2lamda(xone, lamda);
      if (lamda[0] < boxlo[0] || lamda[0] >= boxhi[0] || lamda[1] < boxlo[1] ||
          lamda[1] >= boxhi[1] || lamda[2] < boxlo[2] || lamda[2] >= boxhi[2])
        continue;
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

    // gather reject flags from all of the procs
    int reject_any;
    MPI_Allreduce(&reject, &reject_any, 1, MPI_INT, MPI_MAX, world);
    if (reject_any) continue;

    // all tests passed

    success = true;
    break;
  }

  return success;

}    // void FixSupersaturation::gen_one()

/* ---------------------------------------------------------------------- */
