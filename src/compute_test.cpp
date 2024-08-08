/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_test.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "modify.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeTest::ComputeTest(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg)
{

  scalar_flag = 1;
  vector_flag = 1;
  array_flag = 1;
  size_vector = 10;
  size_array_rows = 10;
  size_array_cols = 10;

  peratom_flag = 1;
  size_peratom_cols = 0;

  local_flag = 1;
  size_local_rows = 10;
  size_local_cols = 10;

  nloc = 1;
}

/* ---------------------------------------------------------------------- */

ComputeTest::~ComputeTest()
{
  if (vector != nullptr) lmp->memory->destroy(vector);
  if (array != nullptr) lmp->memory->destroy(array);
  if (array_atom != nullptr) lmp->memory->destroy(array_atom);
  if (array_local != nullptr) lmp->memory->destroy(array_local);
}

/* ---------------------------------------------------------------------- */

void ComputeTest::init()
{
  if (modify->get_compute_by_style(style).size() > 1)
    if (comm->me == 0) error->warning(FLERR, "More than one compute {}", style);
}

/* ---------------------------------------------------------------------- */

double ComputeTest::compute_scalar()
{
  scalar = 66.6;
  return scalar;
}
void ComputeTest::compute_vector()
{
  if (vector == nullptr) {
    lmp->memory->create(vector, size_vector, "compute:test:vector");
    for (int i = 0; i < size_vector; ++i) { vector[i] = i; }
  }
}
void ComputeTest::compute_array()
{
  if (array == nullptr) {
    lmp->memory->create(array, size_array_rows, size_array_cols, "compute:test:array");
    for (int i = 0; i < size_array_rows; ++i) {
      for (int j = 0; j < size_array_cols; ++j) { array[i][j] = i * j; }
    }
  }
}
void ComputeTest::compute_peratom()
{
  if (atom->nlocal != nloc && array_atom != nullptr) {
    lmp->memory->destroy(array_atom);
    array_atom = nullptr;
  }
  if (array_atom == nullptr) {
    nloc = atom->nlocal;
    lmp->memory->create(array_atom, nloc * size_peratom_cols, "compute:test:array_atom");
    for (int i = 0; i < nloc; ++i) {
      for (int j = 0; j < size_peratom_cols; ++j) { array_atom[i][j] = i * j; }
    }
  }
}
void ComputeTest::compute_local()
{
  if (array_local == nullptr) {
    lmp->memory->create(array_local, size_local_rows, size_local_rows, "compute:test:local_array");
    for (int i = 0; i < size_local_rows; ++i) {
      for (int j = 0; j < size_local_rows; ++j) { array_local[i][j] = i * j; }
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeTest::memory_usage()
{
  double bytes = 0;
  return bytes;
}