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

// TODO: NUCC FILE

#ifdef FIX_CLASS
// clang-format off
FixStyle(langevin/im,FixLangevinIm);
// clang-format on
#else

#ifndef LMP_FIX_LANGEVIN_IM_H
#define LMP_FIX_LANGEVIN_IM_H

#include "fix.h"

namespace LAMMPS_NS {

class FixLangevinIm : public Fix {
 public:
  FixLangevinIm(class LAMMPS *lmp, int narg, char **arg);
  ~FixLangevinIm() noexcept(true) override;
  int setmask() override;
  void init() override;
  void setup(int vflag) override;
  void post_force(int vflag) override;
  void post_force_respa(int vflag, int ilevel, int iloop) override;
  void end_of_step() override;
  void reset_target(double t_new) override;
  void reset_dt() override;
  int modify_param(int narg, char **arg) override;
  double compute_scalar() override;
  double memory_usage() override;
  void *extract(const char *str, int &dim) override;
  void grow_arrays(int nmax) override;
  void copy_arrays(int i, int j, int delflag) override;
  int pack_exchange(int i, double *buf) override;
  int unpack_exchange(int nlocal, double *buf) override;

 protected:
  int nvalues;
  int oflag;
  int tallyflag;
  int zeroflag;
  int tbiasflag{};
  int flangevin_allocated;
  double ascale;
  double t_start;
  double t_stop;
  double t_period;
  double t_target;
  double *gfactor1;
  double *gfactor2;
  double *ratio;
  double energy;
  double energy_onestep{};
  double tsqrt{};
  int tstyle;
  int tvar{};
  char *tstr;

  class AtomVecEllipsoid *avec{};

  int maxatom1;
  int maxatom2;
  double **flangevin;
  double *tforce;
  double **franprev;
  double **lv;    //half step velocity

  char *id_temp;
  class Compute *temperature;

  int nlevels_respa{};
  class RanMars *random;
  int seed;

  template <int Tp_TSTYLEATOM, int Tp_TALLY, int Tp_BIAS, int Tp_RMASS, int Tp_ZERO>
  void post_force_templated();

  void omega_thermostat();
  void angmom_thermostat();
  void compute_target();
};

}    // namespace LAMMPS_NS

#endif
#endif
