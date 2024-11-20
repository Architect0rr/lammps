/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(regen,FixRegen);
// clang-format on
#else

#ifndef LMP_FIX_REGEN_H
#define LMP_FIX_REGEN_H

#include "fix.h"
#include "fix_kedff.h"

namespace LAMMPS_NS {

class FixRegen : public Fix {
 public:
  FixRegen(class LAMMPS *, int, char **);
  ~FixRegen() override;
  int setmask() override;
  void init() override;
  void setup_pre_exchange() override;
  void pre_exchange() override;
  double compute_scalar() override;
  void write_restart(FILE *) override;
  void restart(char *) override;
  void *extract(const char *, int &) override;

  inline bigint get_ninsert() const noexcept(true) { return ninsert; }
  inline bigint get_ninserted() const noexcept(true) { return ninserted; }
  inline void add_ninsert(const bigint n) noexcept(true) { ninsert += n; }
  inline void force_reneigh(const bigint n) noexcept(true) { next_reneighbor = n; }

 private:
  bigint ninsert;
  bigint ninserted;
  int ntype, nfreq, seed, at_once;
  int globalflag{}, localflag{}, maxattempt{}, rateflag{}, scaleflag{}, targetflag{};
  int mode{}, rigidflag{}, shakeflag{}, idnext{}, distflag{}, orientflag{}, warnflag{};
  int varflag{}, vvar{}, xvar{}, yvar{}, zvar{};
  int tempflag{}, groupid{}, markflag{};
  double temperature{}, vsigma{};
  double lo{}, hi{}, deltasq{}, nearsq{}, rate{}, sigma{};
  double vxlo{}, vxhi{}, vylo{}, vyhi{}, vzlo{}, vzhi{};
  double xlo, xhi, ylo, yhi, zlo, zhi, xmid{}, ymid{}, zmid{};
  double rx{}, ry{}, rz{}, tx{}, ty{}, tz{};
  class Region *iregion{};
  char *idregion;
  char *idrigid, *idshake;
  char *vstr{}, *xstr{}, *ystr{}, *zstr{};
  char *xstr_copy{}, *ystr_copy{}, *zstr_copy{};

  FixKedff *fix_keddf = nullptr;

  class Molecule **onemols;
  int nmol{}, natom_max;
  double *molfrac;
  double **coords;
  imageint *imageflags;
  class Fix *fixrigid, *fixshake;
  double oneradius{};

  bigint nfirst;
  tagint maxtag_all{}, maxmol_all{};
  class RanPark *random;

  void find_maxid();
  void options(int, char **);
  int vartest(double, double, double);    // evaluate a variable with new atom position
};

}    // namespace LAMMPS_NS

#endif
#endif
