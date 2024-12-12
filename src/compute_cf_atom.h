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

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(cf/atom,ComputeCFAtom);
// clang-format on
#else

#ifndef COMPUTE_RDF_ATOM_H
#define COMPUTE_RDF_ATOM_H

#include "compute.h"
#include "nucc_cspan.hpp"
#include <array>

namespace LAMMPS_NS {

class ComputeCFAtom : public Compute {
 public:
  ComputeCFAtom(class LAMMPS*, int, char**);
  ~ComputeCFAtom() override;
  void init() override;
  void init_list(int, class NeighList*) override;
  void compute_peratom() override;
  double memory_usage() override;

  inline constexpr const std::array<int, 3>& get_nbins() const noexcept { return nbins; }
  inline constexpr const std::array<NUCC::cspan<double>, 3>& get_bins() const noexcept { return bins; }

 private:
  double ddvol;
  int smooth;
  int nmax;
  class NeighList* list;
  double cutoff;
  double cutsq;

  std::array<double, 3> sigmas;
  std::array<int, 3> nbins;
  std::array<double, 3> norms;
  double norm;

  double** rdf;
  std::array<NUCC::cspan<double>, 3> bins;
  NUCC::cspan<double> rbs;
  NUCC::cspan<double> invdens;
};

}    // namespace LAMMPS_NS

#endif
#endif
