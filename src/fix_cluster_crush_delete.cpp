/*
 ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "fix_cluster_crush_delete.h"
#include "fmt/base.h"
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
#include "memory.h"
#include "modify.h"
#include "update.h"
#include "irregular.h"

#include <cstring>
#include <unordered_map>

using namespace LAMMPS_NS;
using namespace FixConst;

constexpr int DEFAULT_MAXTRY = 1000;
constexpr int DEFAULT_MAXTRY_CALL = 5;
constexpr int MAX_RANDOM_VALUE = 32767;

/* ---------------------------------------------------------------------- */

FixClusterCrushDelete::FixClusterCrushDelete(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), screenflag(1), fileflag(0), scaleflag(0), next_step(0), nloc(0),
    p2m(nullptr), to_restore(0), added_prev(0), at_once(1), fix_temp(false),
    maxtry(::DEFAULT_MAXTRY), maxtry_call(::DEFAULT_MAXTRY_CALL), ntype(0), sigma(0)
{

  restart_pbc = 1;
  pre_exchange_migrate = 1;

  nevery = 1;

  if (narg < 9) { utils::missing_cmd_args(FLERR, "fix cluster/crush", error); }

  // Parse arguments //

  // Target region
  region = domain->get_region_by_id(arg[3]);
  if (region == nullptr) {
    error->all(FLERR, "fix cluster/crush: Cannot find target region {}", arg[3]);
  }

  // Get the critical size
  kmax = utils::inumeric(FLERR, arg[4], true, lmp);
  if (kmax < 2) { error->all(FLERR, "kmax for fix cluster/crush cannot be less than 2"); }

  // Get cluster/size compute
  compute_cluster_size = dynamic_cast<ComputeClusterSize *>(lmp->modify->get_compute_by_id(arg[5]));
  if (compute_cluster_size == nullptr) {
    error->all(FLERR, "fix cluster/crush: Cannot find compute of style 'cluster/size' with id: {}",
               arg[5]);
  }

  // Minimum distance to other atoms from the place atom teleports to
  overlap = utils::numeric(FLERR, arg[6], true, lmp);
  if (overlap < 0) {
    error->all(FLERR, "Minimum distance for fix cluster/crush must be non-negative");
  }

  // apply scaling factor for styles that use distance-dependent factors
  // overlap *= domain->lattice->xlattice;
  odistsq = overlap * overlap;

  // Get the seed for coordinate generator
  int const xseed = utils::inumeric(FLERR, arg[7], true, lmp);
  xrandom = new RanPark(lmp, xseed);

  // Get the ntype atom creation
  ntype = utils::inumeric(FLERR, arg[8], true, lmp);
  if ((ntype <= 0) || (ntype > atom->ntypes)) {
    error->all(FLERR, "Invalid atom type in create_atoms command");
  }

  // Parse optional keywords
  int iarg = 9;
  fp = nullptr;

  while (iarg < narg) {
    if (::strcmp(arg[iarg], "maxtry") == 0) {
      // Max attempts to search for a new suitable location
      maxtry = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      if (maxtry < 1) { error->all(FLERR, "maxtry for fix cluster/crush cannot be less than 1"); }
      iarg += 2;

    } else if (::strcmp(arg[iarg], "temp") == 0) {
      // Monomer temperature
      fix_temp = true;
      monomer_temperature = utils::numeric(FLERR, arg[iarg + 1], true, lmp);
      if (monomer_temperature < 0) {
        error->all(FLERR, "Monomer temperature for fix cluster/crush cannot be negative");
      }

      // Get the seed for velocity generator
      int const vseed = utils::inumeric(FLERR, arg[iarg + 2], true, lmp);
      vrandom = new RanPark(lmp, vseed);
      iarg += 3;

    } else if (::strcmp(arg[iarg], "maxtry_call") == 0) {

      // Get max number of tries for calling delete_monomers()/add_monomers()
      maxtry_call = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      if (maxtry_call < 1) {
        error->all(FLERR, "maxtry_call for fix cluster/crush cannot be less than 1");
      }
      iarg += 2;

    } else if (::strcmp(arg[iarg], "at_once") == 0) {

      // Get max number of tries for calling delete_monomers()/add_monomers()
      at_once = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      if (at_once < 1) { error->all(FLERR, "at_once for fix cluster/crush cannot be less than 1"); }
      iarg += 2;

    } else if (::strcmp(arg[iarg], "noscreen") == 0) {
      // Do not output to screen
      screenflag = 0;
      iarg += 1;

    } else if (::strcmp(arg[iarg], "file") == 0) {
      if (comm->me == 0) {
        // Write output to new file
        fileflag = 1;
        fp = ::fopen(arg[iarg + 1], "w");
        if (fp == nullptr) {
          error->one(FLERR, "Cannot open fix cluster/crush stats file {}: {}", arg[iarg + 1],
                     utils::getsyserror());
        }
      }
      iarg += 2;

    } else if (::strcmp(arg[iarg], "append") == 0) {
      if (comm->me == 0) {
        // Append output to file
        fileflag = 1;
        fp = ::fopen(arg[iarg + 1], "a");
        if (fp == nullptr) {
          error->one(FLERR, "Cannot open fix cluster/crush stats file {}: {}", arg[iarg + 1],
                     utils::getsyserror());
        }
      }
      iarg += 2;

    } else if (::strcmp(arg[iarg], "nevery") == 0) {
      // Get execution period
      nevery = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      iarg += 2;

    } else if (::strcmp(arg[iarg], "units") == 0) {
      if (::strcmp(arg[iarg + 1], "box") == 0) {
        scaleflag = 0;
      } else if (::strcmp(arg[iarg + 1], "lattice") == 0) {
        scaleflag = 1;
      } else {
        error->all(FLERR, "Unknown fix cluster/crush units option {}", arg[iarg + 1]);
      }
      iarg += 2;
    } else {
      error->all(FLERR, "Illegal fix cluster/crush command option {}", arg[iarg]);
    }
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
    error->all(FLERR, "No overlap of box and region for fix cluster/crush");
  }

  if ((comm->me == 0) && (fileflag != 0)) {
    fmt::print(fp, "ntimestep,ntotal,cc,ad,aa,tr\n");
    ::fflush(fp);
  }

  next_step = update->ntimestep - (update->ntimestep % nevery);

  pproc = memory->create(pproc, comm->nprocs * sizeof(int), "cluster/crush:pproc");
  c2c = memory->create(c2c, comm->nprocs * sizeof(int), "cluster/crush:c2c");

  ::memset(xone, 0, 3 * sizeof(double));
  ::memset(lamda, 0, 3 * sizeof(double));

  // Get temp compute
  auto temp_computes = lmp->modify->get_compute_by_style("temp");
  if (temp_computes.empty()) {
    error->all(FLERR, "compute supersaturation: Cannot find compute with style 'temp'.");
  }
  compute_temp = temp_computes[0];
  sigma = std::sqrt(monomer_temperature / atom->mass[ntype]);
}

/* ---------------------------------------------------------------------- */

FixClusterCrushDelete::~FixClusterCrushDelete() noexcept(true)
{
  delete xrandom;
  delete vrandom;

  if ((fp != nullptr) && (comm->me == 0)) {
    ::fflush(fp);
    ::fclose(fp);
  }
  memory->destroy(pproc);
  memory->destroy(c2c);
  if (p2m != nullptr) { memory->destroy(p2m); }
}

/* ---------------------------------------------------------------------- */

void FixClusterCrushDelete::init()
{
  if ((modify->get_fix_by_style(style).size() > 1) && (comm->me == 0)) {
    error->warning(FLERR, "More than one fix {}", style);
  }
}

/* ---------------------------------------------------------------------- */

int FixClusterCrushDelete::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixClusterCrushDelete::pre_exchange()
{
  if (to_restore > 0) {
    postTeleport();
    if (comm->me == 0) { utils::logmesg(lmp, "Restoring..."); }
    int const nloc_prev = atom->nlocal;
    for (int i = 0; (i < at_once) && (to_restore > 0); ++i) {
      if (comm->me == 0) { utils::logmesg(lmp, "Restoring iteration {}...", i); }
      int tries = 0;
      bool succ = false;
      while (!succ) {
        succ = genOneFull();
        if (++tries > maxtry) { break; }
      }
      if (succ) {
        if (coord[0] >= subbonds[0][0] && coord[0] < subbonds[0][1] && coord[1] >= subbonds[1][0] &&
            coord[1] < subbonds[1][1] && coord[2] >= subbonds[2][0] && coord[2] < subbonds[2][1]) {
          atom->avec->create_atom(ntype, xone);
          if (fix_temp) { set_speed(atom->nlocal - 1); }
        }
        --to_restore;
        ++added_prev;
      }
    }
    int added = atom->nlocal > nloc_prev ? 1 : 0;
    int added_any = 0;
    ::MPI_Allreduce(&added,  &added_any, 1, MPI_INT, MPI_MAX, world);
    if (added_any) {
      post_add(nloc_prev);
    }
  }

  if (update->ntimestep < next_step) { return; }
  next_step = update->ntimestep + nevery;

  if (compute_cluster_size->invoked_vector < update->ntimestep - (nevery / 2)) {
    compute_cluster_size->compute_vector();
  }
  const auto cIDs_by_size = compute_cluster_size->cIDs_by_size;
  auto atoms_by_cID = compute_cluster_size->atoms_by_cID;

  if ((nloc < atom->nlocal) && (p2m != nullptr)) { memory->destroy(p2m); }
  if ((nloc < atom->nlocal) || (p2m == nullptr)) {
    nloc = atom->nlocal;
    memory->create(p2m, nloc * sizeof(int), "fix cluster/crush:p2m");
    ::memset(p2m, 0, nloc);
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

  ::memset(c2c, 0, comm->nprocs * sizeof(int));
  c2c[comm->me] = clusters2crush_local;
  ::MPI_Allgather(&clusters2crush_local, 1, MPI_LMP_BIGINT, c2c, 1, MPI_LMP_BIGINT, world);

  ::memset(pproc, 0, comm->nprocs * sizeof(int));
  pproc[comm->me] = atoms2move_local;
  ::MPI_Allgather(&atoms2move_local, 1, MPI_INT, pproc, 1, MPI_INT, world);

  bigint atoms2move_total = 0;
  bigint clusters2crush_total = 0;
  for (int proc = 0; proc < comm->nprocs; ++proc) {
    atoms2move_total += pproc[proc];
    clusters2crush_total += c2c[proc];
  }

  if (clusters2crush_total == 0) {
    if (comm->me == 0) {
      if (screenflag != 0) { utils::logmesg(lmp, "No clusters with size exceeding {}\n", kmax); }
      if (fileflag != 0) {
        fmt::print(fp, "{},{},0,0,{},{}\n", update->ntimestep, atom->natoms, added_prev, to_restore);
        ::fflush(fp);
      }
    }
    return;
  }

  deleteAtoms(atoms2move_local);
  postDelete();
  to_restore += atoms2move_total;

  if (comm->me == 0) {
    // print status
    if (screenflag != 0) {
      utils::logmesg(lmp, "Crushed {} clusters -> deleted {} atoms. Restored: {} atoms\n",
                     clusters2crush_total, atoms2move_total, added_prev);
    }
    if (fileflag != 0) {
      fmt::print(fp, "{},{},{},{},{},{}\n", update->ntimestep, atom->natoms, clusters2crush_total,
                 atoms2move_total, added_prev, to_restore);
      ::fflush(fp);
    }
  }
  added_prev = 0;

}    // void FixClusterCrush::pre_exchange()

/* ---------------------------------------------------------------------- */

void FixClusterCrushDelete::post_add(const int nlocal_previous) noexcept(true)
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

void FixClusterCrushDelete::set_speed(int pID) noexcept(true)
{
  double **v = atom->v;
  // generate velocities
  v[pID][0] = vrandom->gaussian() * sigma;
  v[pID][1] = vrandom->gaussian() * sigma;
  if (domain->dimension == 3) { v[pID][2] = vrandom->gaussian() * sigma; }
}

/* ----------------------------------------------------------------------
  attempts to create coords up to maxtry times
  criteria for insertion: region, triclinic box, overlap
------------------------------------------------------------------------- */

bool FixClusterCrushDelete::genOneFull() noexcept(true)
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

}    // bool FixClusterCrush::gen_one_full()

/* ---------------------------------------------------------------------- */

void FixClusterCrushDelete::postTeleport() noexcept(true)
{
  // move atoms back inside simulation box and to new processors
  // use remap() instead of pbc() in case atoms moved a long distance
  // use irregular() in case atoms moved a long distance

  imageint *image = atom->image;
  for (int i = 0; i < atom->nlocal; ++i) { domain->remap(atom->x[i], image[i]); }

  if (domain->triclinic != 0) { domain->x2lamda(atom->nlocal); }
  domain->reset_box();
  auto *irregular = new Irregular(lmp);
  irregular->migrate_atoms(1);
  delete irregular;
  if (domain->triclinic != 0) { domain->lamda2x(atom->nlocal); }

  // check if any atoms were lost
  bigint nblocal = atom->nlocal;
  bigint natoms = 0;
  ::MPI_Allreduce(&nblocal, &natoms, 1, MPI_LMP_BIGINT, MPI_SUM, world);

  if ((comm->me == 0) && (natoms != atom->natoms)) {
    error->warning(FLERR, "Lost atoms via cluster/crush: original {} current {}", atom->natoms,
                   natoms);
  }
}

/* ---------------------------------------------------------------------- */

void FixClusterCrushDelete::deleteAtoms(int atoms2move_local) noexcept(true)
{
  // delete local atoms
  // reset nlocal

  for (int i = atoms2move_local - 1; i >= 0; --i) {
    atom->avec->copy(atom->nlocal - atoms2move_local + i, p2m[i], 1);
  }

  atom->nlocal -= atoms2move_local;
}    // void FixClusterCrush::delete_monomers(int)

/* ---------------------------------------------------------------------- */

void FixClusterCrushDelete::postDelete() noexcept(true)
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
