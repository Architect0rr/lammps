/*
 ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "fix_cluster_crush.h"

#include <cmath>

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
#include "irregular.h"
#include "lattice.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstring>
#include <unordered_map>

using namespace LAMMPS_NS;
using namespace FixConst;

constexpr int DEFAULT_MAXTRY = 1000;

/* ---------------------------------------------------------------------- */

FixClusterCrush::FixClusterCrush(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), screenflag(1), fileflag(0), velscaleflag(0), velscale(0.0), next_step(0),
    maxtry(DEFAULT_MAXTRY), scaleflag(0), fix_temp(0), nloc(0), p2m(nullptr)
{

  restart_pbc = 1;
  pre_exchange_migrate = 1;

  nevery = 1;

  if (narg < 6) { utils::missing_cmd_args(FLERR, "cluster/crush", error); }

  // Parse arguments //

  // Target region
  region = domain->get_region_by_id(arg[3]);
  if (region == nullptr) { error->all(FLERR, "Cannot find target region {}", arg[3]); }

  // Get the critical size
  kmax = utils::inumeric(FLERR, arg[4], true, lmp);
  if (kmax < 2) { error->all(FLERR, "kmax for cluster/crush cannot be less than 2"); }

  if (strcmp(arg[5], "delete") == 0) {
    teleportflag = 0;
  } else if (strcmp(arg[5], "teleport") == 0) {
    teleportflag = 1;
    if (narg < 9) { utils::missing_cmd_args(FLERR, "cluster/crush", error); }
  } else {
    error->all(FLERR, "Illegal fix cluster/crush keyword: {}", arg[5]);
  }

  // Get cluster/size compute
  // compute_cluster_size = lmp->modify->get_compute_by_id(arg[7]);
  compute_cluster_size = static_cast<ComputeClusterSize *>(lmp->modify->get_compute_by_id(arg[6]));
  if (compute_cluster_size == nullptr) {
    error->all(FLERR, "cluster/crush: Cannot find compute of style 'cluster/size' with id: {}",
               arg[8]);
  }

  if (teleportflag != 0) {
    // Minimum distance to other atoms from the place atom teleports to
    double overlap = utils::numeric(FLERR, arg[7], true, lmp);
    if (overlap < 0) {
      error->all(FLERR, "Minimum distance for fix cluster/crush must be non-negative");
    }

    // apply scaling factor for styles that use distance-dependent factors
    overlap *= domain->lattice->xlattice;
    odistsq = overlap * overlap;

    // Get the seed for coordinate generator
    int const xseed = utils::inumeric(FLERR, arg[8], true, lmp);
    xrandom = new RanPark(lmp, xseed);
  }

  // Parse optional keywords
  int iarg = teleportflag != 0 ? 9 : 7;

  fp = nullptr;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "maxtry") == 0) {
      // Max attempts to search for a new suitable location
      maxtry = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      if (maxtry < 1) { error->all(FLERR, "maxtry for cluster/crush cannot be less than 1"); }
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
      if (comm->me == 0) {
        // Write output to new file
        fileflag = 1;
        fp = fopen(arg[iarg + 1], "w");
        if (fp == nullptr) {
          error->one(FLERR, "Cannot open cluster/crush stats file {}: {}", arg[iarg + 1],
                     utils::getsyserror());
        }
      }
      iarg += 2;

    } else if (strcmp(arg[iarg], "append") == 0) {
      if (comm->me == 0) {
        // Append output to file
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

    } else if (strcmp(arg[iarg], "units") == 0) {
      if (strcmp(arg[iarg + 1], "box") == 0) {
        scaleflag = 0;
      } else if (strcmp(arg[iarg + 1], "lattice") == 0) {
        scaleflag = 1;
      } else {
        error->all(FLERR, "Unknown cluster/crush units option {}", arg[iarg + 1]);
      }
      iarg += 2;
    } else if (strcmp(arg[iarg], "velscale") == 0) {
      velscaleflag = 1;
      velscale = utils::numeric(FLERR, arg[iarg + 1], true, lmp);
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

  if ((region != nullptr) && (region->bboxflag != 0)) {
    xlo = MAX(xlo, region->extent_xlo);
    xhi = MIN(xhi, region->extent_xhi);
    ylo = MAX(ylo, region->extent_ylo);
    yhi = MIN(yhi, region->extent_yhi);
    zlo = MAX(zlo, region->extent_zlo);
    zhi = MIN(zhi, region->extent_zhi);
  }

  if (xlo > xhi || ylo > yhi || zlo > zhi) {
    error->all(FLERR, "No overlap of box and region for cluster/crush");
  }

  if (comm->me == 0 && (fileflag != 0)) {
    fmt::print(fp, "ntimestep,cc,a2m,am,nm,ad\n");
    fflush(fp);
  }

  next_step = update->ntimestep - (update->ntimestep % nevery);

  nprocs = comm->nprocs;
  nptt_rank = memory->create(nptt_rank, nprocs * sizeof(int), "cluster/crush:nptt_rank");
  c2c = memory->create(c2c, nprocs * sizeof(int), "cluster/crush:c2c");

  memset(xone, 0, 3 * sizeof(double));
  memset(lamda, 0, 3 * sizeof(double));

  // Get temp compute
  auto temp_computes = lmp->modify->get_compute_by_style("temp");
  if (temp_computes.empty()) {
    error->all(FLERR, "compute supersaturation: Cannot find compute with style 'temp'.");
  }
  compute_temp = temp_computes[0];
}

/* ---------------------------------------------------------------------- */

FixClusterCrush::~FixClusterCrush() noexcept(true)
{
  delete xrandom;
  delete vrandom;
  if ((fp != nullptr) && (comm->me == 0)) { fclose(fp); }
  memory->destroy(nptt_rank);
  memory->destroy(c2c);
  if (p2m != nullptr) { memory->destroy(p2m); }
}

/* ---------------------------------------------------------------------- */

int FixClusterCrush::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixClusterCrush::pre_exchange()
{
  if (update->ntimestep < next_step) { return; }
  next_step = update->ntimestep + nevery;

  if (compute_cluster_size->invoked_vector < update->ntimestep - (nevery / 2)) {
    compute_cluster_size->compute_vector();
  }
  std::unordered_map<tagint, std::vector<tagint>> const cIDs_by_size =
      compute_cluster_size->cIDs_by_size;
  std::unordered_map<tagint, std::vector<tagint>> atoms_by_cID = compute_cluster_size->atoms_by_cID;

  if (nloc < atom->nlocal && p2m != nullptr) { memory->destroy(p2m); }
  if (nloc < atom->nlocal || p2m == nullptr) {
    nloc = atom->nlocal;
    memory->create(p2m, nloc, "cluster/crush:p2m");
    memset(p2m, 0, nloc);
  }

  // Count amount of local clusters to crush
  bigint clusters2crush_local = 0;
  // Count amount of local atoms to move
  int atoms2move_local = 0;

  for (const auto &[size, cIDs] : cIDs_by_size) {
    if (size > kmax) {
      clusters2crush_local += cIDs.size();
      for (const int cID : cIDs) {
        for (const int pID : atoms_by_cID[cID]) {
          p2m[atoms2move_local] = pID;
          ++atoms2move_local;
        }
      }
    }
  }

  memset(c2c, 0, nprocs * sizeof(int));
  c2c[comm->me] = clusters2crush_local;
  MPI_Allgather(&clusters2crush_local, 1, MPI_LMP_BIGINT, c2c, 1, MPI_LMP_BIGINT, world);

  memset(nptt_rank, 0, nprocs * sizeof(int));
  nptt_rank[comm->me] = atoms2move_local;
  MPI_Allgather(&atoms2move_local, 1, MPI_INT, nptt_rank, 1, MPI_INT, world);

  bigint atoms2move_total = 0;
  bigint clusters2crush_total = 0;
  for (int proc = 0; proc < nprocs; ++proc) {
    atoms2move_total += nptt_rank[proc];
    clusters2crush_total += c2c[proc];
  }

  if (clusters2crush_total == 0) {
    if (comm->me == 0) {
      if (screenflag != 0) { utils::logmesg(lmp, "No clusters with size exceeding {}\n", kmax); }
      if (fileflag != 0) {
        fmt::print(fp, "{},0,0,0,0,0\n", update->ntimestep);
        fflush(fp);
      }
    }
    return;
  }

  bigint not_moved = 0;
  bigint nmoved = 0;

  if (teleportflag != 0) {
    for (int nproc = 0; nproc < nprocs; ++nproc) {
      for (int i = 0; i < nptt_rank[nproc]; ++i) {
        if (gen_one()) {    // if success new coords will be already in xone[]
          ++nmoved;
          if (nproc == comm->me) { set(p2m[i]); }
        } else {
          ++not_moved;
        }
      }
    }
    // warn if did not successfully moved all atoms
    if (nmoved < atoms2move_total) {
      error->warning(FLERR, "Only moved {} atoms out of {} ({}%)", nmoved, atoms2move_total,
                     (100 * nmoved) / atoms2move_total);
    }
    post_teleport();
  } else {
    delete_monomers(atoms2move_local);
    post_delete();
  }

  if (comm->me == 0) {
    // print status
    if (screenflag != 0) {
      utils::logmesg(lmp, "Crushed {} clusters -> {} {} atoms\n", clusters2crush_total,
                     teleportflag != 0 ? "moved" : "deleted",
                     teleportflag != 0 ? nmoved : atoms2move_total);
    }
    if (fileflag != 0) {
      fmt::print(fp, "{},{},{},{},{},{}\n", update->ntimestep, clusters2crush_total,
                 atoms2move_total, nmoved, not_moved, teleportflag != 0 ? 0 : atoms2move_total);
      fflush(fp);
    }
  }

}    // void FixClusterCrush::pre_exchange()

/* ---------------------------------------------------------------------- */

void FixClusterCrush::set(int pID) noexcept(true)
{
  double **x = atom->x;
  double **v = atom->v;

  x[pID][0] = xone[0];
  x[pID][1] = xone[1];
  x[pID][2] = xone[2];

  if (velscaleflag != 0) {
    v[pID][0] *= velscale;
    v[pID][1] *= velscale;
    v[pID][2] *= velscale;
  }

  if (fix_temp != 0) {
    // generate velocities
    constexpr long double c_v = 0.7978845608028653558798921198687L;    // sqrt(2/pi)
    double const sigma = std::sqrt(monomer_temperature / atom->mass[atom->type[pID]]);
    double const v_mean = static_cast<double>(c_v) * sigma;
    v[pID][0] = v_mean + vrandom->gaussian() * sigma;
    v[pID][1] = v_mean + vrandom->gaussian() * sigma;
    if (domain->dimension == 3) { v[pID][2] = v_mean + vrandom->gaussian() * sigma; }
  }
}    // void FixClusterCrush::set(int)

/* ----------------------------------------------------------------------
  attempts to create coords up to maxtry times
  criteria for insertion: region, triclinic box, overlap
------------------------------------------------------------------------- */

bool FixClusterCrush::gen_one() noexcept(true)
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
      double delx = xone[0] - x[i][0];
      double dely = xone[1] - x[i][1];
      double delz = xone[2] - x[i][2];

      const double distsq1 = delx * delx + dely * dely + delz * delz;
      domain->minimum_image(delx, dely, delz);
      const double distsq = delx * delx + dely * dely + delz * delz;
      if (distsq < odistsq || distsq1 < odistsq) {
        reject = 1;
        break;
      }
    }

    // gather reject flags from all of the procs
    int reject_any = 0;
    MPI_Allreduce(&reject, &reject_any, 1, MPI_INT, MPI_MAX, world);
    if (reject_any != 0) { continue; }

    // all tests passed

    success = true;
    break;
  }

  return success;

}    // bool FixClusterCrush::gen_one()

/* ---------------------------------------------------------------------- */

void FixClusterCrush::delete_monomers(int atoms2move_local) noexcept(true)
{
  // delete local atoms
  // reset nlocal

  for (int i = atoms2move_local - 1; i >= 0; --i) {
    atom->avec->copy(atom->nlocal - atoms2move_local + i, p2m[i], 1);
  }

  atom->nlocal -= atoms2move_local;
}    // void FixClusterCrush::delete_monomers(int)

/* ---------------------------------------------------------------------- */

void FixClusterCrush::post_teleport() noexcept(true)
{
  // move atoms back inside simulation box and to new processors
  // use remap() instead of pbc() in case atoms moved a long distance
  // use irregular() in case atoms moved a long distance

  imageint *image = atom->image;
  for (int i = 0; i < atom->nlocal; i++) { domain->remap(atom->x[i], image[i]); }
  // for (int i = 0; i < nptt_rank[comm->me]; i++) {
  //   int pID = p2m[i];
  //   domain->remap(atom->x[pID], image[pID]);
  // }

  if (domain->triclinic != 0) { domain->x2lamda(atom->nlocal); }
  domain->reset_box();
  auto *irregular = new Irregular(lmp);
  irregular->migrate_atoms(1);
  delete irregular;
  if (domain->triclinic != 0) { domain->lamda2x(atom->nlocal); }

  // check if any atoms were lost
  bigint nblocal = atom->nlocal;
  bigint natoms = 0;
  MPI_Allreduce(&nblocal, &natoms, 1, MPI_LMP_BIGINT, MPI_SUM, world);

  if (comm->me == 0 && natoms != atom->natoms) {
    error->warning(FLERR, "Lost atoms via cluster/crush: original {} current {}", atom->natoms,
                   natoms);
  }
}

/* ---------------------------------------------------------------------- */

void FixClusterCrush::post_delete() noexcept(true)
{
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

  const auto *avec_ellipsoid = static_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
  const auto *avec_line = static_cast<AtomVecLine *>(atom->style_match("line"));
  const auto *avec_tri = static_cast<AtomVecTri *>(atom->style_match("tri"));
  const auto *avec_body = static_cast<AtomVecBody *>(atom->style_match("body"));
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
}

/* ---------------------------------------------------------------------- */