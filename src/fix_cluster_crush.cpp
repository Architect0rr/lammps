/*
 ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "fix_cluster_crush.h"
#include <cmath>
#include <fmt/core.h>

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
#include "irregular.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstring>
#include <unordered_map>

using namespace LAMMPS_NS;
using namespace FixConst;

constexpr int DEFAULT_MAXTRY = 1000;
constexpr int DEFAULT_MAXTRY_CALL = 5;
constexpr int MAX_RANDOM_VALUE = 32767;

/* ---------------------------------------------------------------------- */

FixClusterCrush::FixClusterCrush(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), screenflag(1), fileflag(0), scaleflag(0), next_step(0), nloc(0),
    p2m(nullptr), fix_temp(false), velscaleflag(0), velscale(0.0), maxtry(::DEFAULT_MAXTRY),
    algorand(nullptr), maxtry_call(::DEFAULT_MAXTRY_CALL), map(nullptr), ntype(0), sigma(0),
    succ(nullptr)
{

  restart_pbc = 1;
  pre_exchange_migrate = 1;

  nevery = 1;

  if (narg < 6) { utils::missing_cmd_args(FLERR, "fix cluster/crush", error); }

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

  if (::strcmp(arg[6], "delete") == 0) {
    mode = MODE::DELETE;
  } else if (::strcmp(arg[6], "teleport") == 0) {
    mode = MODE::TELEPORT;
    if (narg < 9) { utils::missing_cmd_args(FLERR, "cluster/crush", error); }
  } else if (::strcmp(arg[6], "fastport") == 0) {
    mode = MODE::FASTPORT;
    if (narg < 10) { utils::missing_cmd_args(FLERR, "cluster/crush", error); }
  } else {
    error->all(FLERR, "Illegal fix cluster/crush keyword: {}", arg[6]);
  }

  if ((mode == MODE::TELEPORT) || (mode == MODE::FASTPORT)) {
    // Minimum distance to other atoms from the place atom teleports to
    overlap = utils::numeric(FLERR, arg[7], true, lmp);
    if (overlap < 0) {
      error->all(FLERR, "Minimum distance for fix cluster/crush must be non-negative");
    }

    // apply scaling factor for styles that use distance-dependent factors
    // overlap *= domain->lattice->xlattice;
    odistsq = overlap * overlap;

    // Get the seed for coordinate generator
    int const xseed = utils::inumeric(FLERR, arg[8], true, lmp);
    xrandom = new RanPark(lmp, xseed);
  }

  if (mode == MODE::FASTPORT) {
    // Get the ntype atom creation
    ntype = utils::inumeric(FLERR, arg[9], true, lmp);
    if ((ntype <= 0) || (ntype > atom->ntypes)) {
      error->all(FLERR, "Invalid atom type in create_atoms command");
    }
  }

  // Parse optional keywords
  int iarg = 0;
  if (mode == MODE::DELETE) {
    iarg = 7;
  } else if (mode == MODE::TELEPORT) {
    iarg = 9;
  } else {
    iarg = 10;    // fastport
  }
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
        error->all(FLERR, "maxtry_call for fix supersaturation cannot be less than 1");
      }
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
    } else if (::strcmp(arg[iarg], "velscale") == 0) {
      velscaleflag = 1;
      velscale = utils::numeric(FLERR, arg[iarg + 1], true, lmp);
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
    fmt::print(fp, "ntimestep,ntotal,cc,a2m,am,ad\n");
    ::fflush(fp);
  }

  next_step = update->ntimestep - (update->ntimestep % nevery);

  pproc = memory->create(pproc, comm->nprocs * sizeof(int), "cluster/crush:pproc");
  c2c = memory->create(c2c, comm->nprocs * sizeof(int), "cluster/crush:c2c");

  if (mode == MODE::FASTPORT) {
    ncell[0] = static_cast<bigint>(::floor(subbonds[0][1] - subbonds[0][0]) / overlap);
    ncell[1] = static_cast<bigint>(::floor(subbonds[1][1] - subbonds[1][0]) / overlap);
    ncell[2] = static_cast<bigint>(::floor(subbonds[2][1] - subbonds[2][0]) / overlap);
    map =
        memory->create(map, ncell[0] * ncell[1] * ncell[2] * sizeof(int), "fix_cluster/crush:map");
    if (map == nullptr) { error->all(FLERR, "fix cluster/crush: cannot allocate map"); }
    algorand = new RanPark(lmp, comm->nprocs);
    sigma = std::sqrt(monomer_temperature / atom->mass[ntype]);
    succ = memory->create(succ, comm->nprocs * sizeof(int), "fix cluster/crush:succ");
  }

  ::memset(xone, 0, 3 * sizeof(double));
  ::memset(lamda, 0, 3 * sizeof(double));

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
  delete algorand;

  if ((fp != nullptr) && (comm->me == 0)) {
    ::fflush(fp);
    ::fclose(fp);
  }
  memory->destroy(pproc);
  memory->destroy(c2c);
  if (mode == MODE::FASTPORT) { memory->destroy(map); }
  if (p2m != nullptr) { memory->destroy(p2m); }
}

/* ---------------------------------------------------------------------- */

void FixClusterCrush::init()
{
  if ((modify->get_fix_by_style(style).size() > 1) && (comm->me == 0)) {
    error->warning(FLERR, "More than one fix {}", style);
  }
}

/* ---------------------------------------------------------------------- */

int FixClusterCrush::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

[[nodiscard]] inline bigint FixClusterCrush::i2c(bigint i, bigint j, bigint k) const noexcept(true)
{
  return i + j * ncell[0] + k * ncell[0] * ncell[1];
}

/* ---------------------------------------------------------------------- */

[[nodiscard]] inline bigint FixClusterCrush::x2c(double x, double y, double z) const noexcept(true)
{
  return static_cast<bigint>(x / overlap) + static_cast<bigint>(y / overlap) * ncell[0] +
      static_cast<bigint>(z / overlap) * ncell[0] * ncell[1];
}

/* ---------------------------------------------------------------------- */

void FixClusterCrush::fill_map() noexcept(true)
{
  double **x = atom->x;
  ::memset(map, 0, ncell[0] * ncell[1] * ncell[2] * sizeof(int));
  for (int i = 0; i < atom->nlocal; ++i) {
    ++map[x2c(x[i][0] - subbonds[0][0], x[i][1] - subbonds[1][0], x[i][2] - subbonds[2][0])];
  }
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
        fmt::print(fp, "{},{},0,0,0,0\n", update->ntimestep, atom->natoms);
        ::fflush(fp);
      }
    }
    return;
  }

  bigint const natoms_previous = atom->natoms;
  bigint nmoved = 0;

  if ((mode == MODE::DELETE) || (mode == MODE::FASTPORT)) {
    deleteAtoms(atoms2move_local);
    postDelete();
  } else {
    region->prematch();

    for (int nproc = 0; nproc < comm->nprocs; ++nproc) {
      for (int i = 0; i < pproc[nproc]; ++i) {
        if ((genOneFull()) && (nproc == comm->me)) {
          set(p2m[i]);
          ++nmoved;
        }
      }
    }
    postTeleport();
  }

  if (mode == MODE::FASTPORT) {
    region->prematch();
    int const nlocal_prev = atom->nlocal;
    fill_map();
    add_core();
    post_add(nlocal_prev);

    for (int i = 0; i < comm->nprocs; ++i) { nmoved += pproc[i]; }
    nmoved = atoms2move_total - nmoved;
  }

  if (comm->me == 0) {
    bigint atom_delta = natoms_previous - atom->natoms;
    atom_delta = atom_delta < 0 ? -atom_delta : atom_delta;
    // print status
    if (screenflag != 0) {
      utils::logmesg(lmp, "Crushed {} clusters -> {} {} atoms\n", clusters2crush_total,
                     mode != MODE::DELETE ? "moved" : "deleted", atom_delta);
    }
    if (fileflag != 0) {
      fmt::print(fp, "{},{},{},{},{},{}\n", update->ntimestep, atom->natoms, clusters2crush_total,
                 atoms2move_total, mode != MODE::DELETE ? nmoved : 0,
                 mode == MODE::DELETE ? atom_delta : 0);
      ::fflush(fp);
    }
  }

}    // void FixClusterCrush::pre_exchange()

/* ---------------------------------------------------------------------- */

void FixClusterCrush::add_core() noexcept(true)
{
  bigint sum = 0;
  for (int nproc = 0; nproc < comm->nprocs; ++nproc) { sum += pproc[nproc]; }
  int tries = maxtry_call;

  do {
    pproc[comm->me] = static_cast<int>(sum / comm->nprocs);
    if (static_cast<int>(algorand->uniform() * ::MAX_RANDOM_VALUE) % comm->nprocs == comm->me) {
      pproc[comm->me] += sum % comm->nprocs;
    }

    if (pproc[comm->me] > 0) {
      for (bigint i = 0; (i < ncell[0] - 2) && (pproc[comm->me] > 0); ++i) {
        for (bigint j = 0; (j < ncell[1] - 2) && (pproc[comm->me] > 0); ++j) {
          for (bigint k = 0; (k < ncell[2] - 2) && (pproc[comm->me] > 0); ++k) {
            xone[0] = subbonds[0][0] + (i + 1) * overlap;
            xone[1] = subbonds[1][0] + (j + 1) * overlap;
            xone[2] = subbonds[2][0] + (k + 1) * overlap;
            if ((map[i2c(i, j, k)] == 0) && (map[i2c(i, j + 1, k)] == 0) &&
                (map[i2c(i, j, k + 1)] == 0) && (map[i2c(i, j + 1, k + 1)] == 0) &&
                (map[i2c(i + 1, j, k)] == 0) && (map[i2c(i + 1, j + 1, k)] == 0) &&
                (map[i2c(i + 1, j, k + 1)] == 0) && (map[i2c(i + 1, j + 1, k + 1)] == 0) &&
                (region->match(xone[0], xone[1], xone[2]) != 0)) {
              atom->avec->create_atom(ntype, xone);
              if (fix_temp) { set_speed(atom->nlocal - 1); }
              --pproc[comm->me];
            }
          }
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

/* ---------------------------------------------------------------------- */

void FixClusterCrush::build_tp_map() noexcept(true)
{
  bigint sum = 0;
  for (int nproc = 0; nproc < comm->nprocs; ++nproc) { sum += pproc[nproc]; }
  int tries = maxtry_call;

  do {
    pproc[comm->me] = static_cast<int>(sum / comm->nprocs);
    if (static_cast<int>(algorand->uniform() * ::MAX_RANDOM_VALUE) % comm->nprocs == comm->me) {
      pproc[comm->me] += sum % comm->nprocs;
    }

    if (pproc[comm->me] > 0) {
      for (bigint i = 0; (i < ncell[0] - 1) && (pproc[comm->me] > 0); ++i) {
        for (bigint j = 0; (j < ncell[1] - 1) && (pproc[comm->me] > 0); ++j) {
          for (bigint k = 0; (k < ncell[2] - 1) && (pproc[comm->me] > 0); ++k) {
            xone[0] = (i + 1) * overlap;
            xone[1] = (j + 1) * overlap;
            xone[2] = (k + 1) * overlap;
            if ((map[i2c(i, j, k)] == 0) && (map[i2c(i, j + 1, k)] == 0) &&
                (map[i2c(i, j, k + 1)] == 0) && (map[i2c(i, j + 1, k + 1)] == 0) &&
                (map[i2c(i + 1, j, k)] == 0) && (map[i2c(i + 1, j + 1, k)] == 0) &&
                (map[i2c(i + 1, j, k + 1)] == 0) && (map[i2c(i + 1, j + 1, k + 1)] == 0) &&
                (region->match(xone[0], xone[1], xone[2]) != 0)) {
              atom->avec->create_atom(ntype, xone);
              if (fix_temp) { set_speed(atom->nlocal - 1); }
              --pproc[comm->me];
            }
          }
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

/* ---------------------------------------------------------------------- */

void FixClusterCrush::post_add(const int nlocal_previous) noexcept(true)
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

void FixClusterCrush::set_speed(int pID) noexcept(true)
{
  double **v = atom->v;
  // generate velocities
  v[pID][0] = vrandom->gaussian() * sigma;
  v[pID][1] = vrandom->gaussian() * sigma;
  if (domain->dimension == 3) { v[pID][2] = vrandom->gaussian() * sigma; }
}

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

  if (fix_temp) {
    // generate velocities
    double const sigma = ::sqrt(monomer_temperature / atom->mass[atom->type[pID]]);
    v[pID][0] = vrandom->gaussian() * sigma;
    v[pID][1] = vrandom->gaussian() * sigma;
    if (domain->dimension == 3) { v[pID][2] = vrandom->gaussian() * sigma; }
  }
}    // void FixClusterCrush::set(int)

/* ----------------------------------------------------------------------
  attempts to create coords up to maxtry times
  criteria for insertion: region, triclinic box, overlap
------------------------------------------------------------------------- */

bool FixClusterCrush::genOneFull() noexcept(true)
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

void FixClusterCrush::deleteAtoms(int atoms2move_local) noexcept(true)
{
  // delete local atoms
  // reset nlocal

  for (int i = atoms2move_local - 1; i >= 0; --i) {
    atom->avec->copy(atom->nlocal - atoms2move_local + i, p2m[i], 1);
  }

  atom->nlocal -= atoms2move_local;
}    // void FixClusterCrush::delete_monomers(int)

/* ---------------------------------------------------------------------- */

void FixClusterCrush::postTeleport() noexcept(true)
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

void FixClusterCrush::postDelete() noexcept(true)
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
