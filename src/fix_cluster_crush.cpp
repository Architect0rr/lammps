/*
 ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "fix_cluster_crush.h"
#include "atom.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "modify.h"
#include "memory.h"
#include "update.h"
#include "comm.h"
#include "irregular.h"
#include "lattice.h"
#include "math_special.h"
// #include "input.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

static constexpr double EPSILON = 1.0e-6;
static constexpr int DEFAULT_MAXTRY = 1000;

/* ---------------------------------------------------------------------- */

FixClusterCrush::FixClusterCrush(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg)
{

  restart_pbc = 1;
  fix_temp = 0;
  scaleflag = 0;
  screenflag = 1;
  fileflag = 0;
  maxtry = DEFAULT_MAXTRY;

  if (domain->dimension == 2){
    error->all(FLERR, "cluster/crush is not compatible with 2D yet");
  }

  if (narg < 8) utils::missing_cmd_args(FLERR, "cluster/crush", error);

  // Parse arguments //

  // Target region
  region = domain->get_region_by_id(arg[3]);
  if (region == nullptr){
    error->all(FLERR, "Cannot find target region {}", arg[3]);
  }

  // Minimum distance to other atoms from the place atom teleports to
  double overlap = utils::numeric(FLERR, arg[4], true, lmp);
  if (overlap < 0)
    error->all(FLERR, "Minimum distance for fix cluster/crush must be non-negative");
  // apply scaling factor for styles that use distance-dependent factors

  overlap *= domain->lattice->xlattice;
  odistsq = overlap * overlap;

  // Get the critical size
  kmax = utils::numeric(FLERR, arg[5], true, lmp);
  if (kmax < 2)
    error->all(FLERR, "kmax for cluster/crush cannot be less than 2");

  // Get the seed for coordinate generator
  int xseed = utils::numeric(FLERR, arg[6], true, lmp);
  xrandom = new RanPark(lmp, xseed);

  // Parse optional keywords

  int iarg = 8;
  fp = nullptr;

  while (iarg < narg){
    if (strcmp(arg[iarg], "maxtry") == 0){
      // Max attempts to search for a new suitable location
      maxtry = utils::inumeric(FLERR, arg[iarg + 1], true, lmp);
      if (maxtry < 1)
        error->all(FLERR, "maxtry for cluster/crush cannot be less than 1");
      iarg += 2;

    } else if (strcmp(arg[iarg], "temp") == 0){
      // Monomer temperature
      fix_temp = 1;
      monomer_temperature = utils::numeric(FLERR, arg[iarg + 1], true, lmp);
      if (monomer_temperature < 0)
        error->all(FLERR, "Monomer temperature for cluster/crush cannot be negative");

      // Get the seed for velocity generator
      int vseed = utils::numeric(FLERR, arg[iarg + 2], true, lmp);
      vrandom = new RanPark(lmp, vseed);
      iarg += 3;

    } else if (strcmp(arg[iarg], "noscreen") == 0){
      // Do not output to screen
      screenflag = 0;
      iarg += 1;

    } else if (strcmp(arg[iarg], "file") == 0){
      if (comm->me == 0){
        // Write output to new file
        fileflag = 1;
        fp = fopen(arg[iarg + 1], "w");
        if (fp == nullptr)
          error->one(FLERR, "Cannot open fix print file {}: {}", arg[iarg + 1],
                      utils::getsyserror());
      }
      iarg += 2;

    } else if (strcmp(arg[iarg], "append") == 0){
      if (comm->me == 0){
        // Append output to file
        fileflag = 1;
        fp = fopen(arg[iarg + 1], "a");
        if (fp == nullptr)
          error->one(FLERR, "Cannot open fix print file {}: {}", arg[iarg + 1],
                      utils::getsyserror());
      }
      iarg += 2;

    } else if (strcmp(arg[iarg], "units") == 0) {
      if (strcmp(arg[iarg+1], "box") == 0)
        scaleflag = 0;
      else if (strcmp(arg[iarg+1], "lattice") == 0)
        scaleflag = 1;
      else
        error->all(FLERR, "Unknown cluster/crush units option {}", arg[iarg+1]);
      iarg += 2;

    } else {
      error->all(FLERR, "Illegal cluster/crush command option {}", arg[iarg]);
    }
  }

  // Get cluster/atom compute
  auto computes = lmp->modify->get_compute_by_style("cluster/atom");
  if (computes.empty()){
    error->all(FLERR, "cluster/crush: Cannot find compute with style: 'cluster/atom'");
  }
  compute_cluster_atom = computes.at(0);

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

}

/* ---------------------------------------------------------------------- */

FixClusterCrush::~FixClusterCrush() {
  delete xrandom;
  if (vrandom) delete vrandom;
  if (fp && (comm->me == 0)) fclose(fp);
}

/* ---------------------------------------------------------------------- */

int FixClusterCrush::setmask() {
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixClusterCrush::init() {}

/* ---------------------------------------------------------------------- */

void FixClusterCrush::pre_exchange()
{
  // Do cluster analysis and retrieve data
  compute_cluster_atom->compute_peratom();
  double *cluster_ids = compute_cluster_atom->vector_atom;

  // Clear buffers
  atoms_by_cID.clear();
  cIDs_by_size.clear();

  // Sort atom IDs by cluster IDs
  for (tagint i = 0; i < atom->nlocal; ++i){
    if (atom->mask[i] & groupbit) {
      atoms_by_cID[static_cast<tagint>(cluster_ids[i])].emplace_back(i);
    }
  }

  int clusters2crush_total = 0;
  // Sum cluster size over all procs
  tagint l_size = 0; // local size of cluster
  tagint t_size = 0; // global size of cluster
  for (tagint i = 1; i <= atom->natoms; ++i){
    l_size = 0 ? atoms_by_cID.count(i) == 0 : atoms_by_cID[i].size();
    MPI_Allreduce(&l_size, &t_size, 1, MPI_INT, MPI_SUM, world);
    if (t_size >= kmax){
      ++clusters2crush_total;
      if (l_size > 0){
        cIDs_by_size[t_size].emplace_back(i);
      }
    }
  }

  if (clusters2crush_total == 0){
    if (comm->me == 0){
      if (screenflag) utils::logmesg(lmp, "No clusters with size exceeding {}\n", kmax);
      if (fileflag) {
        fmt::print(fp, "{},{},{}\n", update->ntimestep, 0, 0);
        fflush(fp);
      }
    }
    return;
  }

  // Count amount of local atoms to move
  int num_local_atoms_to_move = 0;

  for (auto [size, v] : cIDs_by_size)
    for (auto cID : v)
      num_local_atoms_to_move += atoms_by_cID[cID].size();

  int nprocs = comm->nprocs;
  int nptt_rank[nprocs];  // number of atoms to move per rank
  memset(nptt_rank, 0, nprocs*sizeof(int));
  nptt_rank[comm->me] = num_local_atoms_to_move;

  MPI_Allgather(&num_local_atoms_to_move, 1, MPI_INT, nptt_rank, 1, MPI_INT, world);

  int atoms2move_total = 0;
  for (int proc = 0; proc < nprocs; ++proc){atoms2move_total += nptt_rank[proc]; }

  // clear global->local map for owned and ghost atoms
  // clear ghost count and any ghost bonus data internal to AtomVec

  // if (atom->map_style != Atom::MAP_NONE) atom->map_clear();
  // atom->nghost = 0;
  // atom->avec->clear_bonus();
  // atom->avec->grow(0);

  double **x = atom->x;
  double **v = atom->v;

  int nmoved = 0;
  for (int nproc = 0; nproc < nprocs; ++nproc){
    if (nproc == comm->me){ //
      for (auto [size, cIDs] : cIDs_by_size){
        for (auto cID : cIDs){
          for (auto pID : atoms_by_cID[cID]){
            if (gen_one()){  // if success new coords will be already in xone[]
              x[pID][0] = xone[0];
              x[pID][1] = xone[1];
              x[pID][2] = xone[2];

              if (fix_temp){
                // generate vellocities
                double vel_mag = 2 * maxwell_distribution3D(vrandom->uniform(), atom->mass[atom->type[pID]], monomer_temperature);
                double vel_mag_sq = vel_mag * vel_mag;
                v[pID][0] = (2*vrandom->uniform()-1)*vel_mag;
                vel_mag_sq -= v[pID][0] * v[pID][0];
                vel_mag = std::sqrt(vel_mag_sq);

                if (domain->dimension == 2){
                  v[pID][1] = vel_mag;
                } else {
                  v[pID][1] = (2*vrandom->uniform()-1)*vel_mag;
                  v[pID][2] = std::sqrt(vel_mag_sq - v[pID][1]*v[pID][1]);
                }
              }

              ++nmoved;
            }
          }
        }
      }
    } else {
      // if it is not me, just keep random gen synchronized and check for overlap
      for (int j = 0; j < nptt_rank[nproc]; ++j){
        gen_one();
      }
    }
  }

  // MPI_Barrier(world);

  // init per-atom fix/compute/variable values for created atoms

  // atom->data_fix_compute_variable(atom->nlocal, atom->nlocal);

  // // set new total # of atoms and error check
  // bigint nblocal = atom->nlocal;
  // MPI_Allreduce(&nblocal, &atom->natoms, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  // if (atom->natoms < 0 || atom->natoms >= MAXBIGINT) error->all(FLERR, "Too many total atoms");

  // // check that atom IDs are valid

  // atom->tag_check();

  // if global map exists, reset it
  // invoke map_init() b/c atom count has grown

  // if (atom->map_style != Atom::MAP_NONE) {
  //   atom->map_init();
  //   atom->map_set();
  // }

  // move atoms back inside simulation box and to new processors
  // use remap() instead of pbc() in case atoms moved a long distance
  // use irregular() in case atoms moved a long distance

  imageint *image = atom->image;
  for (int i = 0; i < atom->nlocal; i++) domain->remap(atom->x[i],image[i]);

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
    error->warning(FLERR, "Lost atoms via cluster/crush: original {} current {}", atom->natoms, natoms);

  int nmoved_total = 0;
  MPI_Allreduce(&nmoved, &nmoved_total, 1, MPI_INT, MPI_SUM, world);

  // warn if did not successfully moved all atoms
  if (nmoved_total < atoms2move_total && comm->me == 0)
    error->warning(
      FLERR,
      "Only moved {} atoms out of {} ({}%)",
      nmoved_total,
      atoms2move_total,
      (100*nmoved_total) / atoms2move_total
      );

  // print status
  if (comm->me == 0) {
    if (screenflag) utils::logmesg(lmp, "Crushed {} clusters -> moved {} atoms\n", clusters2crush_total, nmoved_total);
    if (fileflag) {
      fmt::print(fp, "{},{},{},{}\n", update->ntimestep, clusters2crush_total, nmoved_total, atoms2move_total);
      fflush(fp);
    }
  }

} // void FixClusterCrush::post_integrate()

/* ---------------------------------------------------------------------- */

bool FixClusterCrush::gen_one() {
  // attempt to insert an atom/molecule up to maxtry times
  // criteria for insertion: region, triclinic box, overlap

  int ntry = 0;
  bool success = false;

  while (ntry < maxtry) {
    ++ntry;

    // generate new random position
    xone[0] = xlo + xrandom->uniform() * (xhi - xlo);
    xone[1] = ylo + xrandom->uniform() * (yhi - ylo);
    xone[2] = zlo + xrandom->uniform() * (zhi - zlo);
    if (domain->dimension == 2) xone[2] = 0.0;

    if (region && (region->match(xone[0], xone[1], xone[2]) == 0)) continue;

    if (triclinic) {
      domain->x2lamda(xone, lamda);
      coord = lamda;
      if (coord[0] < boxlo[0] || coord[0] >= boxhi[0] || coord[1] < boxlo[1] ||
          coord[1] >= boxhi[1] || coord[2] < boxlo[2] || coord[2] >= boxhi[2])
        continue;
    } else {
      coord = xone;
    }

    // check for overlap of new atom/mol with all other atoms
    //   including prior insertions
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

} // void FixClusterCrush::gen_one()

/* ----------------------------------------------------------------------
   Assumming input is uniformly distributed, the output has a
   Maxwellian distribution in accordance to given mass and temperature
------------------------------------------------------------------------- */

double FixClusterCrush::maxwell_distribution3D(double prob, double mass, double temperature) noexcept(true) {
  constexpr long double k = 1.4142135623730950488016887242096L; // sqrt(2)
  return k*std::sqrt(temperature/mass)*erfinv(prob);
}

/* ----------------------------------------------------------------------
   Inverse error function
------------------------------------------------------------------------- */

long double FixClusterCrush::erfinv(long double x) noexcept(true) {

  if (x < -1 || x > 1) {
    return NAN;
  } else if (x == 1.0) {
    return INFINITY;
  } else if (x == -1.0) {
    return -INFINITY;
  }

  constexpr long double LN2 = 6.931471805599453094172321214581e-1L;

  constexpr long double A0 = 1.1975323115670912564578e0L;
  constexpr long double A1 = 4.7072688112383978012285e1L;
  constexpr long double A2 = 6.9706266534389598238465e2L;
  constexpr long double A3 = 4.8548868893843886794648e3L;
  constexpr long double A4 = 1.6235862515167575384252e4L;
  constexpr long double A5 = 2.3782041382114385731252e4L;
  constexpr long double A6 = 1.1819493347062294404278e4L;
  constexpr long double A7 = 8.8709406962545514830200e2L;

  constexpr long double B0 = 1.0000000000000000000e0L;
  constexpr long double B1 = 4.2313330701600911252e1L;
  constexpr long double B2 = 6.8718700749205790830e2L;
  constexpr long double B3 = 5.3941960214247511077e3L;
  constexpr long double B4 = 2.1213794301586595867e4L;
  constexpr long double B5 = 3.9307895800092710610e4L;
  constexpr long double B6 = 2.8729085735721942674e4L;
  constexpr long double B7 = 5.2264952788528545610e3L;

  constexpr long double C0 = 1.42343711074968357734e0L;
  constexpr long double C1 = 4.63033784615654529590e0L;
  constexpr long double C2 = 5.76949722146069140550e0L;
  constexpr long double C3 = 3.64784832476320460504e0L;
  constexpr long double C4 = 1.27045825245236838258e0L;
  constexpr long double C5 = 2.41780725177450611770e-1L;
  constexpr long double C6 = 2.27238449892691845833e-2L;
  constexpr long double C7 = 7.74545014278341407640e-4L;

  constexpr long double D0 = 1.4142135623730950488016887e0L;
  constexpr long double D1 = 2.9036514445419946173133295e0L;
  constexpr long double D2 = 2.3707661626024532365971225e0L;
  constexpr long double D3 = 9.7547832001787427186894837e-1L;
  constexpr long double D4 = 2.0945065210512749128288442e-1L;
  constexpr long double D5 = 2.1494160384252876777097297e-2L;
  constexpr long double D6 = 7.7441459065157709165577218e-4L;
  constexpr long double D7 = 1.4859850019840355905497876e-9L;

  constexpr long double E0 = 6.65790464350110377720e0L;
  constexpr long double E1 = 5.46378491116411436990e0L;
  constexpr long double E2 = 1.78482653991729133580e0L;
  constexpr long double E3 = 2.96560571828504891230e-1L;
  constexpr long double E4 = 2.65321895265761230930e-2L;
  constexpr long double E5 = 1.24266094738807843860e-3L;
  constexpr long double E6 = 2.71155556874348757815e-5L;
  constexpr long double E7 = 2.01033439929228813265e-7L;

  constexpr long double F0 = 1.414213562373095048801689e0L;
  constexpr long double F1 = 8.482908416595164588112026e-1L;
  constexpr long double F2 = 1.936480946950659106176712e-1L;
  constexpr long double F3 = 2.103693768272068968719679e-2L;
  constexpr long double F4 = 1.112800997078859844711555e-3L;
  constexpr long double F5 = 2.611088405080593625138020e-5L;
  constexpr long double F6 = 2.010321207683943062279931e-7L;
  constexpr long double F7 = 2.891024605872965461538222e-15L;

  long double abs_x = fabsl(x);

  if (abs_x <= 0.85L) {
    long double r = 0.180625L - 0.25L * x * x;
    long double num = (((((((A7 * r + A6) * r + A5) * r + A4) * r + A3) * r + A2) * r + A1) * r + A0);
    long double den = (((((((B7 * r + B6) * r + B5) * r + B4) * r + B3) * r + B2) * r + B1) * r + B0);
    return x * num / den;
  }

  long double r = sqrtl(LN2 - logl(1.0L - abs_x));

  long double num{}, den{};
  if (r <= 5.0L) {
    r = r - 1.6L;
    num = (((((((C7 * r + C6) * r + C5) * r + C4) * r + C3) * r + C2) * r + C1) * r + C0);
    den = (((((((D7 * r + D6) * r + D5) * r + D4) * r + D3) * r + D2) * r + D1) * r + D0);
  } else {
    r = r - 5.0L;
    num = (((((((E7 * r + E6) * r + E5) * r + E4) * r + E3) * r + E2) * r + E1) * r + E0);
    den = (((((((F7 * r + F6) * r + F5) * r + F4) * r + F3) * r + F2) * r + F1) * r + F0);
  }

  return copysignl(num / den, x);
}

/* ----------------------------------------------------------------------
   Refine inverse error function niter times
------------------------------------------------------------------------- */

long double FixClusterCrush::erfinv_refine(long double x, int niter) noexcept(true) {
  constexpr long double k = 0.8862269254527580136490837416706L; // 0.5 * sqrt(pi)
  long double y = erfinv(x);
  while (niter-- > 0) {
    y -= k * (std::erfl(y) - x) / std::expl(-y * y);
  }
  return y;
}

/* ---------------------------------------------------------------------- */
