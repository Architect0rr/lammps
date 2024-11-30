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

#include "compute_cluster_atom_copy.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <cmath>
#include <map>
#include <vector>

using namespace LAMMPS_NS;

static constexpr int MAXLOOP = 100;

/* ---------------------------------------------------------------------- */

ComputeClusterAtomCopy::ComputeClusterAtomCopy(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), clusterID(nullptr), clusterSize(nullptr)
{
  if (narg != 4) error->all(FLERR, "Illegal compute cluster/atom command");

  double cutoff = utils::numeric(FLERR, arg[3], false, lmp);
  cutsq = cutoff * cutoff;

  peratom_flag = 1;
  size_peratom_cols = 0;
  comm_forward = 1;

  nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputeClusterAtomCopy::~ComputeClusterAtomCopy()
{
  memory->destroy(clusterID);
  memory->destroy(clusterSize);
}

/* ---------------------------------------------------------------------- */

void ComputeClusterAtomCopy::init()
{
  if (atom->tag_enable == 0)
    error->all(FLERR, "Cannot use compute cluster/atom unless atoms have IDs");
  if (force->pair == nullptr)
    error->all(FLERR, "Compute cluster/atom requires a pair style to be defined");
  if (sqrt(cutsq) > force->pair->cutforce)
    error->all(FLERR, "Compute cluster/atom cutoff is longer than pairwise cutoff");

  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);

  if (modify->get_compute_by_style(style).size() > 1)
    if (comm->me == 0) error->warning(FLERR, "More than one compute {}", style);
}

/* ---------------------------------------------------------------------- */

void ComputeClusterAtomCopy::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeClusterAtomCopy::compute_peratom()
{
  int i, j, ii, jj, inum, jnum;
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
  int *ilist, *jlist, *numneigh, **firstneigh;
  double **x = atom->x;

  invoked_peratom = update->ntimestep;

  if (atom->nmax > nmax) {
    memory->destroy(clusterID);
    memory->destroy(clusterSize);
    nmax = atom->nmax;
    memory->create(clusterID, nmax, "cluster/atom:clusterID");
    memory->create(clusterSize, nmax, "cluster/atom:clusterSize");
    vector_atom = clusterID;
  }

  comm->forward_comm();

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  tagint *tag = atom->tag;
  int *mask = atom->mask;

  std::vector<tagint> parent(nmax);
  std::vector<int> size(nmax, 1);

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (mask[i] & groupbit) {
      clusterID[i] = tag[i];
      parent[tag[i]] = tag[i];
      clusterSize[i] = 1;
    } else {
      clusterID[i] = 0;
      clusterSize[i] = 0;
    }
  }

  int change, done, anychange;
  int counter = 0;

  while (counter < MAXLOOP) {
    comm->forward_comm(this);
    ++counter;
    change = 0;
    while (true) {
      done = 1;
      for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        if (!(mask[i] & groupbit)) continue;

        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        jlist = firstneigh[i];
        jnum = numneigh[i];

        for (jj = 0; jj < jnum; jj++) {
          j = jlist[jj];
          j &= NEIGHMASK;
          if (!(mask[j] & groupbit)) continue;

          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx * delx + dely * dely + delz * delz;
          if (rsq < cutsq) {
            tagint rootI = find(clusterID[i], parent);
            tagint rootJ = find(clusterID[j], parent);
            if (rootI != rootJ) {
              union_clusters(rootI, rootJ, parent, size);
              done = 0;
            }
          }
        }
      }
      if (!done) change = 1;
      if (done) break;
    }

    MPI_Allreduce(&change, &anychange, 1, MPI_INT, MPI_MAX, world);
    if (!anychange) break;
  }
  if ((comm->me == 0) && (counter >= MAXLOOP))
    error->warning(FLERR, "Compute cluster/atom did not converge after {} iterations", MAXLOOP);

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (mask[i] & groupbit) {
      clusterID[i] = find(clusterID[i], parent);
      clusterSize[i] = size[clusterID[i]];
    }
  }
}

/* ---------------------------------------------------------------------- */

tagint ComputeClusterAtomCopy::find(tagint x, std::vector<tagint> &parent) {
  if (parent[x] != x) {
    parent[x] = find(parent[x], parent);
  }
  return parent[x];
}

void ComputeClusterAtomCopy::union_clusters(tagint x, tagint y, std::vector<tagint> &parent, std::vector<int> &size) {
  tagint rootX = find(x, parent);
  tagint rootY = find(y, parent);
  if (rootX != rootY) {
    if (size[rootX] < size[rootY]) {
      parent[rootX] = rootY;
      size[rootY] += size[rootX];
    } else {
      parent[rootY] = rootX;
      size[rootX] += size[rootY];
    }
  }
}

/* ---------------------------------------------------------------------- */

int ComputeClusterAtomCopy::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/,
                                              int * /*pbc*/)
{
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = clusterID[j];
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeClusterAtomCopy::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) clusterID[i] = buf[m++];
}

/* ---------------------------------------------------------------------- */

double ComputeClusterAtomCopy::memory_usage()
{
  double bytes = (double) nmax * (sizeof(double) + sizeof(int));
  return bytes;
}
