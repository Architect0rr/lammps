/*
 ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "fix_cluster_crush.h"

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

#include <cmath>
#include <cstring>
#include <unordered_map>

using namespace LAMMPS_NS;
using namespace FixConst;

constexpr int DEFAULT_MAXTRY = 1000;

/* ---------------------------------------------------------------------- */

FixClusterCrush::FixClusterCrush(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{

  restart_pbc = 1;
  fix_temp = 0;
  scaleflag = 0;
  screenflag = 1;
  fileflag = 0;
  maxtry = DEFAULT_MAXTRY;
  next_step = 0;
  nevery = 1;

  if (domain->dimension == 2) { error->all(FLERR, "cluster/crush is not compatible with 2D yet"); }

  if (narg < 9) utils::missing_cmd_args(FLERR, "cluster/crush", error);

  // Parse arguments //

  // Target region
  region = domain->get_region_by_id(arg[3]);
  if (region == nullptr) { error->all(FLERR, "Cannot find target region {}", arg[3]); }

  // Minimum distance to other atoms from the place atom teleports to
  double overlap = utils::numeric(FLERR, arg[4], true, lmp);
  if (overlap < 0) error->all(FLERR, "Minimum distance for fix cluster/crush must be non-negative");

  // apply scaling factor for styles that use distance-dependent factors
  overlap *= domain->lattice->xlattice;
  odistsq = overlap * overlap;

  // Get the critical size
  kmax = utils::numeric(FLERR, arg[5], true, lmp);
  if (kmax < 2) error->all(FLERR, "kmax for cluster/crush cannot be less than 2");

  // Get the seed for coordinate generator
  int xseed = utils::numeric(FLERR, arg[6], true, lmp);
  xrandom = new RanPark(lmp, xseed);

  // Get cluster/size compute
  // compute_cluster_size = lmp->modify->get_compute_by_id(arg[7]);
  compute_cluster_size = static_cast<ComputeClusterSize *>(lmp->modify->get_compute_by_id(arg[7]));
  if (compute_cluster_size == nullptr) {
    error->all(FLERR, "cluster/crush: Cannot find compute of style 'cluster/size' with id: {}",
               arg[7]);
  }

  // Parse optional keywords

  int iarg = 8;
  fp = nullptr;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "maxtry") == 0) {
      // Max attempts to search for a new suitable location
      maxtry = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      if (maxtry < 1) error->all(FLERR, "maxtry for cluster/crush cannot be less than 1");
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
      if (comm->me == 0) {
        // Write output to new file
        fileflag = 1;
        fp = fopen(arg[iarg + 1], "w");
        if (fp == nullptr)
          error->one(FLERR, "Cannot open fix print file {}: {}", arg[iarg + 1],
                     utils::getsyserror());
      }
      iarg += 2;

    } else if (strcmp(arg[iarg], "append") == 0) {
      if (comm->me == 0) {
        // Append output to file
        fileflag = 1;
        fp = fopen(arg[iarg + 1], "a");
        if (fp == nullptr)
          error->one(FLERR, "Cannot open fix print file {}: {}", arg[iarg + 1],
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

  if (comm->me == 0 && fileflag) {
    fmt::print(fp, "ntimestep,cc,pm,p2m\n");
    fflush(fp);
  }

  next_step = update->ntimestep - (update->ntimestep % nevery);

  nprocs = comm->nprocs;
  memory->create(nptt_rank, nprocs * sizeof(int), "cluster/crush:nptt_rank");

  // memory->create(xone, static_cast<int>(3 * sizeof(double)), "cluster/crush:xone");
  // memory->create(lamda, static_cast<int>(3 * sizeof(double)), "cluster/crush:lambda");

  memset(xone, 0, 3 * sizeof(double));
  memset(lamda, 0, 3 * sizeof(double));
}

/* ---------------------------------------------------------------------- */

FixClusterCrush::~FixClusterCrush()
{
  delete xrandom;
  if (vrandom) delete vrandom;
  if (fp && (comm->me == 0)) fclose(fp);
  memory->destroy(nptt_rank);
}

/* ---------------------------------------------------------------------- */

int FixClusterCrush::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixClusterCrush::init() {}

/* ---------------------------------------------------------------------- */

void FixClusterCrush::pre_exchange()
{
  if (update->ntimestep < next_step) return;
  next_step = update->ntimestep + nevery;

  if (compute_cluster_size->invoked_vector < update->ntimestep - (nevery / 2)) {
    compute_cluster_size->compute_vector();
  }
  auto cIDs_by_size = compute_cluster_size->cIDs_by_size;
  auto atoms_by_cID = compute_cluster_size->atoms_by_cID;

  int clusters2crush_total = 0;
  for (const auto &[size, clids] : cIDs_by_size) {
    if (size > kmax) { clusters2crush_total += clids.size(); }
  }

  if (clusters2crush_total == 0) {
    if (comm->me == 0) {
      if (screenflag) utils::logmesg(lmp, "No clusters with size exceeding {}\n", kmax);
      if (fileflag) {
        fmt::print(fp, "{},{},{}, {}\n", update->ntimestep, 0, 0, 0);
        fflush(fp);
      }
    }
    return;
  }

  // Count amount of local atoms to move
  int num_local_atoms_to_move = 0;

  for (auto [size, v] : cIDs_by_size)
    for (auto cID : v) num_local_atoms_to_move += atoms_by_cID[cID].size();

  memset(nptt_rank, 0, nprocs * sizeof(int));
  nptt_rank[comm->me] = num_local_atoms_to_move;

  MPI_Allgather(&num_local_atoms_to_move, 1, MPI_INT, nptt_rank, 1, MPI_INT, world);

  int atoms2move_total = 0;
  for (int proc = 0; proc < nprocs; ++proc) { atoms2move_total += nptt_rank[proc]; }

  double **x = atom->x;
  double **v = atom->v;

  int nmoved = 0;
  for (int nproc = 0; nproc < nprocs; ++nproc) {
    if (nproc == comm->me) {    //
      for (auto [size, cIDs] : cIDs_by_size) {
        for (auto cID : cIDs) {
          for (auto pID : atoms_by_cID[cID]) {
            if (gen_one()) {    // if success new coords will be already in xone[]
              x[pID][0] = xone[0];
              x[pID][1] = xone[1];
              x[pID][2] = xone[2];

              if (fix_temp) {
                // generate vellocities
                constexpr long double c_v = 0.7978845608028653558798921198687L;    // sqrt(2/pi)
                double sigma = std::sqrt(monomer_temperature / atom->mass[atom->type[pID]]);
                double v_mean = c_v * sigma;
                v[pID][0] = v_mean + vrandom->gaussian() * sigma;
                v[pID][1] = v_mean + vrandom->gaussian() * sigma;
                if (domain->dimension == 3) { v[pID][2] = v_mean + vrandom->gaussian() * sigma; }
              }

              ++nmoved;
            }
          }
        }
      }
    } else {
      // if it is not me, just keep random gen synchronized and check for overlap
      for (int j = 0; j < nptt_rank[nproc]; ++j) { gen_one(); }
    }
  }

  // move atoms back inside simulation box and to new processors
  // use remap() instead of pbc() in case atoms moved a long distance
  // use irregular() in case atoms moved a long distance

  imageint *image = atom->image;
  for (int i = 0; i < atom->nlocal; i++) domain->remap(atom->x[i], image[i]);

  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->reset_box();
  auto irregular = new Irregular(lmp);
  irregular->migrate_atoms(1);
  delete irregular;
  if (domain->triclinic) domain->lamda2x(atom->nlocal);

  // check if any atoms were lost
  bigint nblocal = atom->nlocal;
  bigint natoms = 0;
  MPI_Allreduce(&nblocal, &natoms, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  if (natoms != atom->natoms && comm->me == 0)
    error->warning(FLERR, "Lost atoms via cluster/crush: original {} current {}", atom->natoms,
                   natoms);

  int nmoved_total = 0;
  MPI_Allreduce(&nmoved, &nmoved_total, 1, MPI_INT, MPI_SUM, world);

  // warn if did not successfully moved all atoms
  if (nmoved_total < atoms2move_total && comm->me == 0)
    error->warning(FLERR, "Only moved {} atoms out of {} ({}%)", nmoved_total, atoms2move_total,
                   (100 * nmoved_total) / atoms2move_total);

  // print status
  if (comm->me == 0) {
    if (screenflag)
      utils::logmesg(lmp, "Crushed {} clusters -> moved {} atoms\n", clusters2crush_total,
                     nmoved_total);
    if (fileflag) {
      fmt::print(fp, "{},{},{},{}\n", update->ntimestep, clusters2crush_total, nmoved_total,
                 atoms2move_total);
      fflush(fp);
    }
  }

}    // void FixClusterCrush::post_integrate()

/* ----------------------------------------------------------------------
  attempts to create coords up to maxtry times
  criteria for insertion: region, triclinic box, overlap
------------------------------------------------------------------------- */

bool FixClusterCrush::gen_one()
{

  int ntry = 0;
  bool success = false;

  while (ntry < maxtry) {
    ++ntry;

    // generate new random position
    xone[0] = xlo + xrandom->uniform() * (xhi - xlo);
    xone[1] = ylo + xrandom->uniform() * (yhi - ylo);
    if (domain->dimension == 3) { xone[2] = zlo + xrandom->uniform() * (zhi - zlo); }

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
    int nlocal = atom->nlocal;

    double delx, dely, delz, distsq;
    int reject = 0;

    // check new position for overlapping with all local atoms
    for (int i = 0; i < nlocal; i++) {
      delx = xone[0] - x[i][0];
      dely = xone[1] - x[i][1];
      delz = xone[2] - x[i][2];
      domain->minimum_image(delx, dely, delz);
      distsq = delx * delx + dely * dely + delz * delz;
      if (distsq < odistsq) {
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

}    // void FixClusterCrush::gen_one()

/* ---------------------------------------------------------------------- */