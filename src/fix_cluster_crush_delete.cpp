/*
 ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "fix_cluster_crush_delete.h"
#include "compute_cluster_size_ext.h"
#include "compute_cluster_temps.h"
#include "nucc_cspan.hpp"

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
#include "fmt/base.h"
#include "group.h"
#include "input.h"
#include "lattice.h"
#include "memory.h"
#include "modify.h"
#include "random_park.h"
#include "region.h"
#include "update.h"
#include "variable.h"

#include <cmath>
#include <cstring>
#include <unordered_map>

using namespace LAMMPS_NS;
using namespace FixConst;

constexpr int DEFAULT_MAXTRY = 1000;

/* ---------------------------------------------------------------------- */

FixClusterCrushDelete::FixClusterCrushDelete(LAMMPS* lmp, int narg, char** arg) : Fix(lmp, narg, arg)
{
  restart_pbc = 1;
  nevery      = 1;

  if (narg < 9) { utils::missing_cmd_args(FLERR, "fix cluster/crush/delete", error); }

  // Parse arguments //

  // Target region
  region = domain->get_region_by_id(arg[3]);
  if (region == nullptr) { error->all(FLERR, "{}: Cannot find target region {}", style, arg[3]); }
  if (region->bboxflag == 0) { error->all(FLERR, "{}: region does not support a bounding box", style); }
  // if (region->dynamic_check() != 0) { error->all(FLERR,"{}: region cannot be dynamic", style); }

  // Get the critical size
  kmax = utils::inumeric(FLERR, arg[4], true, lmp);
  if (kmax < 2) { error->all(FLERR, "{}: kmax cannot be less than 2", style); }

  // Get cluster/size compute
  compute_cluster_size = dynamic_cast<ComputeClusterSizeExt*>(lmp->modify->get_compute_by_id(arg[5]));
  if (compute_cluster_size == nullptr) { error->all(FLERR, "{}: Cannot find compute of style 'cluster/size' with id: {}", style, arg[5]); }
  if (kmax > compute_cluster_size->get_size_cutoff()) {
    error->all(FLERR, "{}: kmax cannot be bigger than its value of compute size/cluster", style);
  }

  // Minimum distance to other atoms from the place atom teleports to
  overlap = utils::numeric(FLERR, arg[6], true, lmp);
  if (overlap < 0) { error->all(FLERR, "{}: Minimum distance must be non-negative", style); }
  overlapsq = overlap * overlap;

  // Get the seed for coordinate generator
  int xseed = utils::inumeric(FLERR, arg[7], true, lmp);
  xrandom   = new RanPark(lmp, xseed);

  // Get the seed for coordinate generator
  int vseed = utils::inumeric(FLERR, arg[8], true, lmp);
  vrandom   = new RanPark(lmp, xseed);

  // Get the ntype atom creation
  ntype     = utils::inumeric(FLERR, arg[9], true, lmp);
  if ((ntype <= 0) || (ntype > atom->ntypes)) { error->all(FLERR, "{}: nvalid atom type in create_atoms command", style); }

  // Parse optional keywords
  int iarg = 10;
  fp       = nullptr;

  while (iarg < narg) {
    if (::strcmp(arg[iarg], "maxtry") == 0) {
      // Max attempts to search for a new suitable location
      maxtry = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      if (maxtry < 1) { error->all(FLERR, "{}: maxtry cannot be less than 1", style); }
      iarg += 2;

    } else if (::strcmp(arg[iarg], "temp") == 0) {
      // Monomer temperature
      fix_temp            = true;
      monomer_temperature = utils::numeric(FLERR, arg[iarg + 1], true, lmp);
      if (monomer_temperature < 0) { error->all(FLERR, "{}: Monomer temperature cannot be negative", style); }
      vdist = DIST::DIST_GAUSSIAN;
      iarg += 2;

    } else if (::strcmp(arg[iarg], "noscreen") == 0) {
      // Do not output to screen
      screenflag = 0;
      iarg += 1;

    } else if (::strcmp(arg[iarg], "file") == 0) {
      if (comm->me == 0) {
        // Write output to new file
        fileflag = 1;
        fp       = ::fopen(arg[iarg + 1], "w");
        if (fp == nullptr) { error->one(FLERR, "{}: Cannot open stats file {}: {}", style, arg[iarg + 1], utils::getsyserror()); }
      }
      iarg += 2;
    } else if (::strcmp(arg[iarg], "append") == 0) {
      if (comm->me == 0) {
        // Append output to file
        fileflag = 1;
        fp       = ::fopen(arg[iarg + 1], "a");
        if (fp == nullptr) { error->one(FLERR, "{}: Cannot open stats file {}: {}", style, arg[iarg + 1], utils::getsyserror()); }
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
        error->all(FLERR, "{}: Unknown units option {}", style, arg[iarg + 1]);
      }
      iarg += 2;

    } else if (strcmp(arg[iarg], "var") == 0) {
      if (iarg + 2 > narg) { utils::missing_cmd_args(FLERR, "fix deposit var", error); }
      vstr    = utils::strdup(arg[iarg + 1]);
      varflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg], "set") == 0) {
      if (iarg + 3 > narg) { utils::missing_cmd_args(FLERR, "fix deposit set", error); }
      if (strcmp(arg[iarg + 1], "x") == 0) {
        xstr = utils::strdup(arg[iarg + 2]);
      } else if (strcmp(arg[iarg + 1], "y") == 0) {
        ystr = utils::strdup(arg[iarg + 2]);
      } else if (strcmp(arg[iarg + 1], "z") == 0) {
        zstr = utils::strdup(arg[iarg + 2]);
      } else {
        error->all(FLERR, "Unknown fix deposit set option {}", arg[iarg + 2]);
      }
      iarg += 3;
    } else if (strcmp(arg[iarg], "group") == 0) {
      if (iarg + 2 > narg) { error->all(FLERR, "Illegal fix deposit command"); }
      groupid = group->find(arg[iarg + 1]);
      if (groupid <= 0) { error->all(FLERR, "Specified group not found or group all is used"); }
      iarg += 2;
    } else if (strcmp(arg[iarg], "maxtry") == 0) {
      if (iarg + 2 > narg) { error->all(FLERR, "Illegal fix deposit command"); }
      maxtry = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "vx") == 0) {
      if (iarg + 3 > narg) { error->all(FLERR, "Illegal fix deposit command"); }
      vels[0] = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      vels[1] = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      iarg += 3;
    } else if (strcmp(arg[iarg], "vy") == 0) {
      if (iarg + 3 > narg) { error->all(FLERR, "Illegal fix deposit command"); }
      vels[2] = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      vels[3] = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      iarg += 3;
    } else if (strcmp(arg[iarg], "vz") == 0) {
      if (iarg + 3 > narg) { error->all(FLERR, "Illegal fix deposit command"); }
      vels[4] = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      vels[5] = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      iarg += 3;
    } else if (strcmp(arg[iarg], "units") == 0) {
      if (iarg + 2 > narg) { error->all(FLERR, "Illegal fix deposit command"); }
      if (strcmp(arg[iarg + 1], "box") == 0) {
        scaleflag = 0;
      } else if (strcmp(arg[iarg + 1], "lattice") == 0) {
        scaleflag = 1;
      } else {
        error->all(FLERR, "Illegal fix deposit command");
      }
      iarg += 2;
    } else if (strcmp(arg[iarg], "inpoint") == 0) {
      if (iarg + 5 > narg) { error->all(FLERR, "Illegal fix deposit command"); }
      xmid[0] = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      xmid[1] = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      xmid[2] = utils::numeric(FLERR, arg[iarg + 3], false, lmp);
      xsigma  = utils::numeric(FLERR, arg[iarg + 4], false, lmp);
      xdist   = DIST::DIST_GAUSSIAN;
      iarg += 5;
    } else {
      error->all(FLERR, "{}: Illegal command option {}", style, arg[iarg]);
    }

    // Get temp compute
    auto temp_computes = lmp->modify->get_compute_by_style("temp");
    if (temp_computes.empty()) { error->all(FLERR, "{}: Cannot find compute with style 'temp'.", style); }
    compute_temp = dynamic_cast<ComputeClusterTemp*>(temp_computes[0]);
    if (atom->mass_setflag[ntype] == 0) { error->all(FLERR, "{}: Atom mass for atom type {} is not set!", style, ntype); }
    vsigma    = ::sqrt(monomer_temperature / atom->mass[ntype]);

    sbonds[0] = region->extent_xlo;
    sbonds[1] = region->extent_xhi;
    sbonds[2] = region->extent_ylo;
    sbonds[3] = region->extent_yhi;
    sbonds[4] = region->extent_zlo;
    sbonds[5] = region->extent_zhi;

    if (domain->triclinic == 0) {
      if (sbonds[0] < domain->boxlo[0] || sbonds[1] > domain->boxhi[0] || sbonds[2] < domain->boxlo[1] || sbonds[3] > domain->boxhi[1] ||
          sbonds[4] < domain->boxlo[2] || sbonds[5] > domain->boxhi[2]) {
        error->all(FLERR, "Deposition region extends outside simulation box");
      }
    } else {
      if (sbonds[0] < domain->boxlo_bound[0] || sbonds[1] > domain->boxhi_bound[0] || sbonds[2] < domain->boxlo_bound[1] ||
          sbonds[3] > domain->boxhi_bound[1] || sbonds[4] < domain->boxlo_bound[2] || sbonds[5] > domain->boxhi_bound[2]) {
        error->all(FLERR, "Deposition region extends outside simulation box");
      }
    }

    // error check and further setup for mode = MOLECULE

    if (atom->tag_enable == 0) { error->all(FLERR, "Cannot use fix_deposit unless atoms have IDs"); }

    // apply scaling factor for styles that use distance-dependent factors
    // setup scaling

    if (scaleflag != 0) {
      double xscale = domain->lattice->xlattice;
      double yscale = domain->lattice->ylattice;
      double zscale = domain->lattice->zlattice;
      // apply scaling to all input parameters with dist/vel units

      overlap *= xscale;
      overlapsq *= xscale * xscale;
      vels[0] *= xscale;
      vels[1] *= xscale;
      vels[2] *= yscale;
      vels[3] *= yscale;
      vels[4] *= zscale;
      vels[5] *= zscale;
      xmid[0] *= xscale;
      xmid[1] *= yscale;
      xmid[2] *= zscale;
      xsigma *= ::pow(xscale * yscale * zscale, 1. / 3.);    // same as in region sphere
      vsigma *= ::pow(xscale * yscale * zscale, 1. / 3.);    // same as in region sphere
    }
  }

  // error check and further setup for variable test

  if ((vstr == nullptr) && ((xstr != nullptr) || (ystr != nullptr) || (zstr != nullptr))) {
    error->all(FLERR, "Incomplete use of variables in fix deposit command");
  }
  if ((vstr != nullptr) && ((xstr == nullptr) && (ystr == nullptr) && (zstr == nullptr))) {
    error->all(FLERR, "Incomplete use of variables in fix deposit command");
  }

  if (varflag != 0) {
    vars[3] = input->variable->find(vstr);
    if (vars[3] < 0) { error->all(FLERR, "Variable {} for fix deposit does not exist", vstr); }
    if (input->variable->equalstyle(vars[3]) == 0) { error->all(FLERR, "Variable for fix deposit is invalid style"); }

    if (xstr != nullptr) {
      vars[0] = input->variable->find(xstr);
      if (vars[0] < 0) { error->all(FLERR, "Variable {} for fix deposit does not exist", xstr); }
      if (input->variable->internalstyle(vars[0]) == 0) { error->all(FLERR, "Variable for fix deposit is invalid style"); }
    }
    if (ystr != nullptr) {
      vars[1] = input->variable->find(ystr);
      if (vars[1] < 0) { error->all(FLERR, "Variable {} for fix deposit does not exist", ystr); }
      if (input->variable->internalstyle(vars[1]) == 0) { error->all(FLERR, "Variable for fix deposit is invalid style"); }
    }
    if (zstr != nullptr) {
      vars[2] = input->variable->find(zstr);
      if (vars[2] < 0) { error->all(FLERR, "Variable {} for fix deposit does not exist", zstr); }
      if (input->variable->internalstyle(vars[2]) == 0) { error->all(FLERR, "Variable for fix deposit is invalid style"); }
    }
  }

  if ((comm->me == 0) && (fileflag != 0)) {
    fmt::print(fp, "ntimestep,ntotal,cc,ad,added,tr\n");
    ::fflush(fp);
  }
}

/* ---------------------------------------------------------------------- */

FixClusterCrushDelete::~FixClusterCrushDelete() noexcept(true)
{
  if ((fp != nullptr) && (comm->me == 0)) {
    ::fflush(fp);
    ::fclose(fp);
  }
  pproc.destroy(memory);
  c2c.destroy(memory);
  p2m.destroy(memory);

  delete xrandom;
  delete vrandom;
  delete[] vstr;
  delete[] xstr;
  delete[] ystr;
  delete[] zstr;
}

/* ---------------------------------------------------------------------- */

void FixClusterCrushDelete::init()
{
  if ((modify->get_fix_by_style(style).size() > 1) && (comm->me == 0)) { error->warning(FLERR, "More than one fix {}", style); }

  pproc.create(memory, comm->nprocs, "cluster/crush/delete:pproc");
  c2c.create(memory, comm->nprocs, "cluster/crush/delete:c2c");

  nloc = atom->nlocal;
  p2m.create(memory, nloc, "cluster/crush/delete:p2m");
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
  if (update->ntimestep < next_step) { return; }
  next_step = update->ntimestep + nevery;

  if (compute_cluster_size->invoked_vector < update->ntimestep) { compute_cluster_size->compute_vector(); }
  const auto& cIDs_by_size = *compute_cluster_size->get_cIDs_by_size();

  if (nloc < atom->nlocal) {
    nloc = atom->nlocal;
    p2m.grow(memory, nloc, "cluster/crush/delete:p2m");
    p2m.reset();
  }

  // Count amount of local clusters to crush
  int clusters2crush_local = 0;
  // Count amount of local atoms to move
  int atoms2move_local     = 0;

  int nclusters            = dynamic_cast<ComputeClusterSizeExt*>(compute_cluster_size)->get_cluster_map()->size();
  const auto& clusters     = dynamic_cast<ComputeClusterSizeExt*>(compute_cluster_size)->get_clusters();
  for (int i = 0; i < nclusters; ++i) {
    const auto& clstr = clusters[i];
    if (clstr.g_size > kmax) {
      ++clusters2crush_local;
      ::memcpy(p2m.data(), clstr.atoms().data(), clstr.l_size * sizeof(int));
      atoms2move_local += clstr.l_size;
    }
  }

  c2c.reset();
  c2c[comm->me] = clusters2crush_local;
  ::MPI_Allgather(&clusters2crush_local, 1, MPI_INT, c2c.data(), 1, MPI_INT, world);

  pproc.reset();
  pproc[comm->me] = atoms2move_local;
  ::MPI_Allgather(&atoms2move_local, 1, MPI_INT, pproc.data(), 1, MPI_INT, world);

  int atoms2move_total     = 0;
  int clusters2crush_total = 0;
  for (int proc = 0; proc < comm->nprocs; ++proc) {
    atoms2move_total += pproc[proc];
    clusters2crush_total += c2c[proc];
  }

  if (clusters2crush_total > 0) {
    deleteAtoms(atoms2move_local);
    postDelete();
  }

  to_insert += atoms2move_total;
  int to_insert_prev = to_insert;

  add();

  if (comm->me == 0) {
    // print status
    if (screenflag != 0) { utils::logmesg(lmp, "Crushed {} clusters -> deleted {} atoms.\n", clusters2crush_total, atoms2move_total); }
    if (fileflag != 0) {
      fmt::print(fp, "{},{},{},{},{},{}\n", update->ntimestep, atom->natoms, clusters2crush_total, atoms2move_total, to_insert_prev - to_insert,
                 to_insert);
      ::fflush(fp);
    }
  }

}    // void FixClusterCrush::pre_exchange()

/* ---------------------------------------------------------------------- */

void FixClusterCrushDelete::deleteAtoms(int atoms2move_local) noexcept(true)
{
  // delete local atoms
  // reset nlocal

  for (int i = atoms2move_local - 1; i >= 0; --i) { atom->avec->copy(atom->nlocal - atoms2move_local + i, p2m[i], 1); }

  atom->nlocal -= atoms2move_local;
}    // void FixClusterCrush::delete_monomers(int)

/* ---------------------------------------------------------------------- */

void FixClusterCrushDelete::add()
{
  int warnflag = 0;
  double coord[3];
  double r[3];
  double vnew[3];

  // clear ghost count (and atom map) and any ghost bonus data
  //   internal to AtomVec
  // same logic as beginning of Comm::exchange()
  // do it now b/c inserting atoms will overwrite ghost atoms

  if (atom->map_style != Atom::MAP_NONE) { atom->map_clear(); }
  atom->nghost = 0;
  atom->avec->clear_bonus();

  double* sublo = nullptr;
  double* subhi = nullptr;
  if (domain->triclinic == 0) {
    sublo = domain->sublo;
    subhi = domain->subhi;
  } else {
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  }

  // find maxid in case other fixes deleted.inserted atoms

  tagint* tag = atom->tag;

  tagint max  = 0;
  for (int i = 0; i < atom->nlocal; i++) { max = MAX(max, tag[i]); }
  tagint maxtag_all = 0;
  MPI_Allreduce(&max, &maxtag_all, 1, MPI_LMP_TAGINT, MPI_MAX, world);

  region->prematch();

  int ninserted = 0;
  for (int added = 0; added < to_insert; ++added) {
    // attempt an insertion until successful

    int success = 0;
    int attempt = 0;
    while (attempt < maxtry) {
      ++attempt;

      // choose random position for new particle within region
      if (xdist == DIST::DIST_UNIFORM) {
        do {
          coord[0] = sbonds[0] + xrandom->uniform() * (sbonds[1] - sbonds[0]);
          coord[1] = sbonds[2] + xrandom->uniform() * (sbonds[2] - sbonds[2]);
          coord[2] = sbonds[4] + xrandom->uniform() * (sbonds[5] - sbonds[4]);
        } while (region->match(coord[0], coord[1], coord[2]) == 0);
      } else if (xdist == DIST::DIST_GAUSSIAN) {
        do {
          coord[0] = xmid[0] + xrandom->gaussian() * xsigma;
          coord[1] = xmid[1] + xrandom->gaussian() * xsigma;
          coord[2] = xmid[2] + xrandom->gaussian() * xsigma;
        } while (region->match(coord[0], coord[1], coord[2]) == 0);
      } else {
        error->all(FLERR, "Unknown particle distribution in fix deposit");
      }

      if ((varflag != 0) && vartest(coord[0], coord[1], coord[2]) == 0) { continue; }

      if (check_overlap(coord) != 1) { continue; }

      int nlocalprev = atom->nlocal;

      generate_velocity(vnew);

      double* newcoord;
      double lamda[3];
      if (domain->triclinic != 0) {
        domain->x2lamda(coord, lamda);
        newcoord = lamda;
      } else {
        newcoord = coord;
      }

      int proceed = newcoord[0] >= sublo[0] && newcoord[0] < subhi[0] && newcoord[1] >= sublo[1] && newcoord[1] < subhi[1] &&
          newcoord[2] >= sublo[2] && newcoord[2] < subhi[2];
      int proceed_any = 0;
      ::MPI_Allreduce(&proceed, &proceed_any, 1, MPI_INT, MPI_SUM, world);
      if ((proceed_any > 1) && (comm->me == 0)) { error->warning(FLERR, "{}: More than one places atom", style); }
      if (proceed != 0) {
        atom->avec->create_atom(ntype, coord);
        int n          = atom->nlocal - 1;
        atom->tag[n]   = maxtag_all + 1;
        atom->mask[n]  = 1 | groupbit;
        atom->image[n] = (static_cast<imageint>(IMGMAX) << IMG2BITS) | (static_cast<imageint>(IMGMAX) << IMGBITS) | IMGMAX;
        atom->v[n][0]  = vnew[0];
        atom->v[n][1]  = vnew[1];
        atom->v[n][2]  = vnew[2];
        modify->create_attribute(n);

        if (groupid != 0) { atom->mask[n] |= (1 << groupid); }
      }

      success = 1;
      break;
    }

    // warn if not successful b/c too many attempts

    if ((warnflag != 0) && (success == 0) && comm->me == 0) {
      error->warning(FLERR, "One or more particle depositions were unsuccessful");
      warnflag = 0;
    }

    if (success != 0) {
      ++atom->natoms;
      ++maxtag_all;
      ++ninserted;
      if (atom->natoms < 0) { error->all(FLERR, "Too many total atoms"); }
      if (maxtag_all >= MAXTAGINT) { error->all(FLERR, "New atom IDs exceed maximum allowed ID"); }
    }
  }

  to_insert -= ninserted;

  atom->tag_extend();
  atom->tag_check();

  // rebuild atom map

  if (atom->map_style != Atom::MAP_NONE) {
    atom->map_init();
    atom->map_set();
  }
}

/* ---------------------------------------------------------------------- */

bool FixClusterCrushDelete::check_overlap(double* coord)
{
  double** x       = atom->x;
  const int nlocal = atom->nlocal;

  int flag         = 0;
  for (int i = 0; i < nlocal; i++) {
    double delx = coord[0] - x[i][0];
    double dely = coord[1] - x[i][1];
    double delz = coord[2] - x[i][2];
    domain->minimum_image(delx, dely, delz);
    double rsq = delx * delx + dely * dely + delz * delz;
    if (rsq < overlapsq) {
      flag = 1;
      break;
    }
  }
  int flagall = 0;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_MAX, world);
  return flagall == 0;
}

/* ---------------------------------------------------------------------- */

void FixClusterCrushDelete::generate_velocity(double* vnew)
{
  if (fix_temp != 0) {
    vnew[0] = vrandom->gaussian() * vsigma;
    vnew[1] = vrandom->gaussian() * vsigma;
    vnew[2] = vrandom->gaussian() * vsigma;
  } else {
    vnew[0] = vels[0] + vrandom->uniform() * (vels[1] - vels[0]);
    vnew[1] = vels[2] + vrandom->uniform() * (vels[3] - vels[2]);
    vnew[2] = vels[4] + vrandom->uniform() * (vels[5] - vels[4]);
  }
}

/* ---------------------------------------------------------------------- */

void FixClusterCrushDelete::postDelete() noexcept(true)
{
  if (atom->molecular == Atom::ATOMIC) {
    tagint* tag      = atom->tag;
    int const nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; ++i) { tag[i] = 0; }
    atom->tag_extend();
  }

  // reset atom->natoms and also topology counts

  bigint nblocal = atom->nlocal;
  ::MPI_Allreduce(&nblocal, &atom->natoms, 1, MPI_INT, MPI_SUM, world);

  // reset bonus data counts

  const auto* avec_ellipsoid = dynamic_cast<AtomVecEllipsoid*>(atom->style_match("ellipsoid"));
  const auto* avec_line      = dynamic_cast<AtomVecLine*>(atom->style_match("line"));
  const auto* avec_tri       = dynamic_cast<AtomVecTri*>(atom->style_match("tri"));
  const auto* avec_body      = dynamic_cast<AtomVecBody*>(atom->style_match("body"));
  bigint nlocal_bonus        = 0;

  if (atom->nellipsoids > 0) {
    nlocal_bonus = avec_ellipsoid->nlocal_bonus;
    ::MPI_Allreduce(&nlocal_bonus, &atom->nellipsoids, 1, MPI_INT, MPI_SUM, world);
  }
  if (atom->nlines > 0) {
    nlocal_bonus = avec_line->nlocal_bonus;
    ::MPI_Allreduce(&nlocal_bonus, &atom->nlines, 1, MPI_INT, MPI_SUM, world);
  }
  if (atom->ntris > 0) {
    nlocal_bonus = avec_tri->nlocal_bonus;
    ::MPI_Allreduce(&nlocal_bonus, &atom->ntris, 1, MPI_INT, MPI_SUM, world);
  }
  if (atom->nbodies > 0) {
    nlocal_bonus = avec_body->nlocal_bonus;
    ::MPI_Allreduce(&nlocal_bonus, &atom->nbodies, 1, MPI_INT, MPI_SUM, world);
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

/* ----------------------------------------------------------------------
   test a generated atom position against variable evaluation
   first set x,y,z values in internal variables
------------------------------------------------------------------------- */

int FixClusterCrushDelete::vartest(double x, double y, double z)
{
  if (xstr != nullptr) { input->variable->internal_set(vars[0], x); }
  if (ystr != nullptr) { input->variable->internal_set(vars[1], y); }
  if (zstr != nullptr) { input->variable->internal_set(vars[2], z); }

  double const value = input->variable->compute_equal(vars[3]);

  if (value == 0.0) { return 0; }
  return 1;
}
