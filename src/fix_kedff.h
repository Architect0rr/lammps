/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(kedff,FixKedff);
// clang-format on
#else

#ifndef LAMMPS_FIX_KEDFF_H
#define LAMMPS_FIX_KEDFF_H

#include "atom.h"
#include "compute.h"
#include "fix.h"
#include "memory.h"
#include "update.h"

namespace LAMMPS_NS {

class FixKedff : public Fix {
 public:
  FixKedff(class LAMMPS *lmp, int narg, char **arg);
  ~FixKedff() noexcept(true) override;
  int setmask() override;
  void end_of_step() override;

  inline void grow_arrays(int nmax) override { memory->grow(vstore, nmax, "store:vstore"); }
  inline void copy_arrays(int i, int j, int /*delflag*/) override { vstore[j] = vstore[i]; }
  int pack_border(int, int *, double *) override;
  int unpack_border(int, int, double *) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_restart(int, double *) override;
  void unpack_restart(int, int) override;
  inline int size_restart(int /*nlocal*/) override { return 2; }
  inline int maxsize_restart() override { return 2; }

  inline double memory_usage() override { return atom->nmax * sizeof(int) + 4 * sizeof(double); }
  inline void mark(const int n) noexcept(true) { vstore[n] = update->ntimestep; }
  inline void unmark(const int n) noexcept(true) { vstore[n] = 0; }
  bigint invoked_endofstep;
  double *engs_global = nullptr;

 protected:
  int delay;
  Compute *compute_ke_atom = nullptr;
  double *engs = nullptr;

  int *vstore = nullptr;    // vector storage
};

}    // namespace LAMMPS_NS

#endif
#endif
