/*
 ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "fix_capture.h"

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
#include "modify.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <unordered_map>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixCapture::FixCapture(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), action(ACTION::COUNT), screenflag(true), fileflag(false)
{

  restart_pbc = 1;

  if (narg < 5) { utils::missing_cmd_args(FLERR, "fix capture", error); }

  // Parse arguments //

  // Target region
  region = domain->get_region_by_id(arg[3]);
  if (region == nullptr) { error->all(FLERR, "fix capture: Cannot find target region {}", arg[3]); }

  // Get number of sigmas
  nsigma = utils::inumeric(FLERR, arg[4], true, lmp);

  int iarg = 5;
  fp = nullptr;

  while (iarg < narg) {
    if (::strcmp(arg[iarg], "action") == 0) {

      if (::strcmp(arg[iarg + 1], "count") == 0) {

        action = ACTION::COUNT;
        iarg += 2;

      } else if (::strcmp(arg[iarg + 1], "delete") == 0) {

        action = ACTION::DELETE;
        iarg += 2;

      } else if (::strcmp(arg[iarg + 1], "cool") == 0) {

        action = ACTION::SLOW;
        // Get the seed for velocity generator
        int const vseed = utils::inumeric(FLERR, arg[iarg + 2], true, lmp);
        vrandom = new RanPark(lmp, vseed);
        iarg += 3;

      } else {
        error->all(FLERR, "Unknown mode for fix capture: {}", arg[iarg + 1]);
      }

    } else if (::strcmp(arg[iarg], "noscreen") == 0) {

      // Do not output to screen
      screenflag = false;
      iarg += 1;

    } else if (::strcmp(arg[iarg], "file") == 0) {

      // Write output to new file
      if (comm->me == 0) {
        fileflag = true;
        fp = ::fopen(arg[iarg + 1], "w");
        if (fp == nullptr) {
          error->one(FLERR, "Cannot open fix capture stats file {}: {}", arg[iarg + 1],
                     utils::getsyserror());
        }
      }
      iarg += 2;

    } else if (::strcmp(arg[iarg], "append") == 0) {

      // Append output to file
      if (comm->me == 0) {
        fileflag = true;
        fp = ::fopen(arg[iarg + 1], "a");
        if (fp == nullptr) {
          error->one(FLERR, "Cannot open fix capture stats file {}: {}", arg[iarg + 1],
                     utils::getsyserror());
        }
      }
      iarg += 2;
    } else {
      error->all(FLERR, "Illegal fix capture command");
    }
  }

  // Get temp compute
  auto temp_computes = lmp->modify->get_compute_by_style("temp");
  if (temp_computes.empty()) {
    error->all(FLERR, "fix capture: Cannot find compute with style 'temp'.");
  }
  compute_temp = temp_computes[0];

  if (fileflag && (comm->me == 0)) {
    fmt::print(fp, "ntimestep,n\n");
    ::fflush(fp);
  }
}

/* ---------------------------------------------------------------------- */

FixCapture::~FixCapture() noexcept(true)
{
  delete vrandom;

  if ((fp != nullptr) && (comm->me == 0)) {
    ::fflush(fp);
    ::fclose(fp);
  }
}

/* ---------------------------------------------------------------------- */

int FixCapture::setmask()
{
  int mask = 0;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCapture::init()
{
  if ((modify->get_fix_by_style(style).size() > 1) && (comm->me == 0)) {
    error->warning(FLERR, "More than one fix {}", style);
  }

  typeids.clear();
  typeids.reserve(atom->ntypes);
  for (int i = 0; i < atom->nlocal; ++i) { typeids.try_emplace(atom->type[i], 0.0, 0.0); }
  for (const auto &[k, v] : typeids) {
    if (atom->mass_setflag[k] == 0) {
      error->all(FLERR, "fix capture: mass is not set for atom type {}.", k);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixCapture::final_integrate()
{
  if (compute_temp->invoked_scalar != update->ntimestep) { compute_temp->compute_scalar(); }

  constexpr long double c_v = 1.4142135623730950488016887242097L;    // sqrt(2)
  for (auto &[k, v] : typeids) {
    v.first = ::sqrt(compute_temp->scalar / atom->mass[k]);
    v.second = static_cast<double>(c_v) * v.first;
  }

  double **v = atom->v;

  bigint ncaptured_local = 0;
  for (int i = 0; i < atom->nlocal; ++i) {
    const auto &[sigma, vmean] = typeids[atom->type[i]];
    if (((atom->mask[i] & groupbit) != 0) &&
        (region->match(atom->x[i][0], atom->x[i][1], atom->x[i][2]) != 0) &&
        (::sqrt(v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]) >
         (vmean + nsigma * sigma))) {
      if (action == ACTION::SLOW) {
        v[i][0] = (vmean + vrandom->gaussian() * sigma) / 2;
        v[i][1] = (vmean + vrandom->gaussian() * sigma) / 2;
        v[i][2] = (vmean + vrandom->gaussian() * sigma) / 2;
      } else if (action == ACTION::DELETE) {
        atom->avec->copy(atom->nlocal - 1, i, 1);
        --atom->nlocal;
        --i;
      } else {
        // for linter
      }

      ++ncaptured_local;
    }
  }

  if ((action == ACTION::DELETE) && (ncaptured_local > 0)) { post_delete(); }

  bigint ncaptured_total = 0;
  ::MPI_Allreduce(&ncaptured_local, &ncaptured_total, 1, MPI_LMP_BIGINT, MPI_SUM, world);

  if (fileflag && (comm->me == 0)) {
    fmt::print(fp, "{},{}\n", update->ntimestep, ncaptured_total);
    ::fflush(fp);
  }
}

/* ---------------------------------------------------------------------- */

void FixCapture::check_overlap() noexcept(true)
{
  constexpr long double a_v = 0.8 * 1.0220217810393767580226573302752L;
  constexpr long double b_v = 0.1546370863640482533333333333333L;
  auto const rl = static_cast<double>(a_v) *
      ::exp(static_cast<double>(b_v) * ::pow(compute_temp->scalar, 2.791206046910478));

  bigint nclose_local = 0;
  double **x = atom->x;
  for (int i = 0; i < atom->nlocal; ++i) {
    for (int j = i + 1; j < atom->nmax; ++j) {
      const double dx = x[i][0] - x[j][0];
      const double dy = x[i][1] - x[j][1];
      const double dz = x[i][2] - x[j][2];
      if (dx * dx + dy * dy + dz * dz < rl * rl) { ++nclose_local; }
    }
  }

  bigint nclose_total = 0;
  ::MPI_Allreduce(&nclose_local, &nclose_total, 1, MPI_LMP_BIGINT, MPI_SUM, world);
}

/* ---------------------------------------------------------------------- */

void FixCapture::post_delete() noexcept(true)
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
