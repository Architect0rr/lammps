// /* -*- c++ -*- ----------------------------------------------------------
//    LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
//    https://www.lammps.org/, Sandia National Laboratories
//    LAMMPS development team: developers@lammps.org

//    Copyright (2003) Sandia Corporation.  Under the terms of Contract
//    DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
//    certain rights in this software.  This software is distributed under
//    the GNU General Public License.

//    See the README file in the top-level LAMMPS directory.
// ------------------------------------------------------------------------- */

// #ifdef COMPUTE_CLASS
// // clang-format off
// ComputeStyle(cluster/atom/copy,ComputeClusterAtomCopy);
// // clang-format on
// #else

// #ifndef LMP_COMPUTE_CLUSTER_ATOM_COPY_H
// #define LMP_COMPUTE_CLUSTER_ATOM_COPY_H

// #include "compute.h"
// #include <vector>

// namespace LAMMPS_NS {

// class ComputeClusterAtomCopy : public Compute {
//  public:
//   ComputeClusterAtomCopy(class LAMMPS *, int, char **);
//   ~ComputeClusterAtomCopy() override;
//   void init() override;
//   void init_list(int, class NeighList *) override;
//   void compute_peratom() override;
//   int pack_forward_comm(int, int *, double *, int, int *) override;
//   void unpack_forward_comm(int, int, double *) override;
//   double memory_usage() override;

//  private:
//   int nmax;
//   double cutsq;
//   class NeighList *list;
//   double *clusterID;
//   int *clusterSize;

//   tagint find(tagint, std::vector<tagint> &);
//   void union_clusters(tagint, tagint, std::vector<tagint> &, std::vector<int> &);
// };

// }    // namespace LAMMPS_NS

// #endif
// #endif
