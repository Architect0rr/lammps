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
#include "fix_regen.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstring>
#include <unordered_map>

using namespace LAMMPS_NS;
using namespace FixConst;

constexpr int DEFAULT_MAXTRY = 1000;

/* ---------------------------------------------------------------------- */

FixClusterCrushDelete::FixClusterCrushDelete(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), screenflag(1), fileflag(0), scaleflag(0), next_step(0), nloc(0),
    p2m(nullptr), at_once(1), groupname(std::string()), fix_temp(false), maxtry(::DEFAULT_MAXTRY),
    ntype(0), sigma(0), reneigh_forced(false), ninserted_prev(0)
{
  restart_pbc = 1;

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

  // Get the seed for coordinate generator
  xseed = utils::inumeric(FLERR, arg[7], true, lmp);

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

      iarg += 2;    // 3

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
    } else if (::strcmp(arg[iarg], "group") == 0) {
      groupname = arg[iarg + 1];

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

  if ((comm->me == 0) && (fileflag != 0)) {
    fmt::print(fp, "ntimestep,ntotal,cc,ad,added,tr\n");
    ::fflush(fp);
  }

  next_step = update->ntimestep - (update->ntimestep % nevery);

  memory->create(pproc, comm->nprocs * sizeof(int), "cluster/crush:pproc");
  memory->create(c2c, comm->nprocs * sizeof(int), "cluster/crush:c2c");

  // Get temp compute
  auto temp_computes = lmp->modify->get_compute_by_style("temp");
  if (temp_computes.empty()) {
    error->all(FLERR, "compute supersaturation: Cannot find compute with style 'temp'.");
  }
  compute_temp = temp_computes[0];
  if (atom->mass_setflag[ntype] == 0) {
    error->all(FLERR, "Fix cluster/crush_delete: Atom mass for atom type {} is not set!", ntype);
  }
  sigma = ::sqrt(monomer_temperature / atom->mass[ntype]);
}

/* ---------------------------------------------------------------------- */

FixClusterCrushDelete::~FixClusterCrushDelete() noexcept(true)
{
  if ((fp != nullptr) && (comm->me == 0)) {
    ::fflush(fp);
    ::fclose(fp);
  }
  if (pproc != nullptr) { memory->destroy(pproc); }
  if (c2c != nullptr) { memory->destroy(c2c); }
  if (p2m != nullptr) { memory->destroy(p2m); }
}

/* ---------------------------------------------------------------------- */

void FixClusterCrushDelete::init()
{
  if ((modify->get_fix_by_style(style).size() > 1) && (comm->me == 0)) {
    error->warning(FLERR, "More than one fix {}", style);
  }

  std::string fixcmd =
      fmt::format("CCDREGENFIX all regen 1 {} 1 {} {} region {} near {} "
                  "attempt {} temp {} units box",
                  ntype, xseed, at_once, region->id, overlap, maxtry, monomer_temperature);
  if (groupname.length() > 0) { fixcmd += fmt::format(" group {}", groupname); }
  fix_regen = dynamic_cast<FixRegen *>(modify->add_fix(fixcmd));
  fix_regen->init();
}

/* ---------------------------------------------------------------------- */

int FixClusterCrushDelete::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

[[gnu::hot]] void FixClusterCrushDelete::pre_exchange()
{
  if (reneigh_forced) {
    next_reneighbor = 0;
    reneigh_forced = false;
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
        bigint ninserted = fix_regen->get_ninserted();
        fmt::print(fp, "{},{},0,0,{},{}\n", update->ntimestep, atom->natoms,
                   ninserted - ninserted_prev, fix_regen->get_ninsert() - ninserted);
        ::fflush(fp);
      }
    }
    ninserted_prev = fix_regen->get_ninserted();
    return;
  }

  deleteAtoms(atoms2move_local);
  postDelete();
  next_reneighbor = update->ntimestep + 1;
  reneigh_forced = true;
  fix_regen->add_ninsert(atoms2move_total);
  fix_regen->force_reneigh(next_reneighbor);

  if (comm->me == 0) {
    // print status
    if (screenflag != 0) {
      utils::logmesg(lmp, "Crushed {} clusters -> deleted {} atoms.\n", clusters2crush_total,
                     atoms2move_total);
    }
    if (fileflag != 0) {
      bigint ninserted = fix_regen->get_ninserted();
      fmt::print(fp, "{},{},{},{},{},{}\n", update->ntimestep, atom->natoms, clusters2crush_total,
                 atoms2move_total, ninserted - ninserted_prev,
                 fix_regen->get_ninsert() - ninserted);
      ::fflush(fp);
    }
  }

  ninserted_prev = fix_regen->get_ninserted();

}    // void FixClusterCrush::pre_exchange()

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
