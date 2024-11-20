/*
 ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "fix_kedff.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixKedff::FixKedff(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg), invoked_endofstep(0)
{
  vector_flag = 1;
  size_vector = 2;
  extvector = 0;
  dynamic_group_allow = 1;

  if (narg < 3) { error->all(FLERR, "Illegal compute ke/mono command"); }

  delay = utils::inumeric(FLERR, arg[3], true, lmp);
  if (delay < 1) { error->all(FLERR, "Compute {}: delay cannot be less than 1", style); }

  // Get ke/atom compute
  auto computes = lmp->modify->get_compute_by_style("ke/atom");
  if (computes.empty()) {
    error->all(FLERR, "compute {}: Cannot find compute with style 'ke/atom'", style);
  }
  compute_ke_atom = computes[0];

  memory->create(engs, 2, "compute_ke_mono:engs");
  memory->create(engs_global, 2, "compute_ke_mono:engs_global");

  grow_arrays(atom->nmax);
  ::memset(vstore, 0, atom->nmax * sizeof(int));
  atom->add_callback(Atom::GROW);
  // atom->add_callback(Atom::RESTART);
  // atom->add_callback(Atom::BORDER);
}

/* ---------------------------------------------------------------------- */

FixKedff::~FixKedff() noexcept(true)
{
  // unregister callbacks to this fix from Atom class
  atom->delete_callback(id, Atom::GROW);
  // atom->delete_callback(id, Atom::RESTART);
  // atom->delete_callback(id, Atom::BORDER);
  if (vstore != nullptr) { memory->destroy(vstore); }

  if (engs != nullptr) { memory->destroy(engs); }
  if (engs_global != nullptr) { memory->destroy(engs_global); }
}

/* ---------------------------------------------------------------------- */

int FixKedff::setmask()
{
  int mask = 0;
  // mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixKedff::end_of_step()
{
  invoked_endofstep = update->ntimestep;
  if (compute_ke_atom->invoked_peratom != update->ntimestep) { compute_ke_atom->compute_peratom(); }
  ::memset(engs, 0.0, 2 * sizeof(double));
  ::memset(engs_global, 0.0, 2 * sizeof(double));

  const double *const kes = compute_ke_atom->vector_atom;
  int marked = 0;

  for (int i = 0; i < atom->nlocal; ++i) {
    if ((atom->mask[i] & groupbit) != 0) {
      if (update->ntimestep - vstore[i] < delay) {
        engs[0] += kes[i];
        ++marked;
      } else {
        engs[1] += kes[i];
      }
    }
  }

  engs[0] /= marked;
  engs[1] /= group->count(igroup) - marked;

  ::MPI_Allreduce(engs, engs_global, 2, MPI_DOUBLE, MPI_SUM, world);
  engs[0] /= comm->nprocs;
  engs[1] /= comm->nprocs;
}

/* ----------------------------------------------------------------------
   pack values for border communication at re-neighboring
------------------------------------------------------------------------- */

int FixKedff::pack_border(int n, int *list, double *buf)
{
  int j{};
  int m = 0;
  for (int i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = vstore[j];
  }

  return m;
}

/* ----------------------------------------------------------------------
   unpack values for border communication at re-neighboring
------------------------------------------------------------------------- */

int FixKedff::unpack_border(int n, int first, double *buf)
{
  int m = 0;
  int const last = first + n;
  for (int i = first; i < last; i++) { vstore[i] = buf[m++]; }

  return m;
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixKedff::pack_exchange(int i, double *buf)
{
  buf[0] = vstore[i];

  return 1;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixKedff::unpack_exchange(int nlocal, double *buf)
{
  vstore[nlocal] = buf[0];

  return 1;
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for restart file
------------------------------------------------------------------------- */

int FixKedff::pack_restart(int i, double *buf)
{
  // pack buf[0] this way because other fixes unpack it
  buf[0] = 2;
  buf[1] = vstore[i];

  return 2;
}

/* ----------------------------------------------------------------------
   unpack values from atom->extra array to restart the fix
------------------------------------------------------------------------- */

void FixKedff::unpack_restart(int nlocal, int nth)
{
  double **extra = atom->extra;

  // skip to Nth set of extra values
  // unpack the Nth first values this way because other fixes pack them

  int m = 0;
  for (int i = 0; i < nth; i++) { m += static_cast<int>(extra[nlocal][m]); }
  m++;

  vstore[nlocal] = extra[nlocal][m];
}
