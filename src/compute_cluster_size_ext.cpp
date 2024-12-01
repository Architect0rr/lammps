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

#include "compute_cluster_size_ext.h"
#include <utility>

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;

#define ComputeClusterSizeIm_ALLOC_COEFF 1.2

/* ---------------------------------------------------------------------- */

ComputeClusterSizeExt::ComputeClusterSizeExt(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), nloc(0), dist(nullptr), nc_global(0), natom_loc(0)
{
  vector_flag = 1;
  extvector = 0;
  size_vector = 0;
  size_vector_variable = 1;

  if (comm->nprocs > LMP_NUCC_CLUSTER_MAX_OWNERS) {
    error->all(
        FLERR,
        "Number of processor exceeds MAX_OWNER limit. Recompile with higher MAX_OWNER limit.");
  }

  if (narg < 4) { utils::missing_cmd_args(FLERR, "compute cluster/size", error); }

  // Parse arguments //

  // Get cluster/atom compute
  compute_cluster_atom = lmp->modify->get_compute_by_id(arg[3]);
  if (compute_cluster_atom == nullptr) {
    error->all(
        FLERR,
        "compute cluster/size: Cannot find compute with style 'cluster/atom' with given id: {}",
        arg[3]);
  }

  // Get the critical size
  size_cutoff = utils::inumeric(FLERR, arg[4], true, lmp);
  if (size_cutoff < 1) {
    error->all(FLERR, "size_cutoff for compute cluster/size must be greater than 0");
  }

  // alloc = new CustomAllocator<std::pair<const int, cluster_ptr>>(1024, memory, keeper);
  // cluster_map =
  //   new std::u
  // keeper = new MemoryKeeper(memory);
  // alloc = new CustomAllocator<std::pair<const int, cluster_ptr>>(1024, memory, keeper);
  // cluster_map =
  //   new std::unordered_map<int, cluster_ptr, std::hash<int>, std::equal_to<int>, CustomAllocator<std::pair<const int, cluster_ptr>>>(*alloc);

  size_vector = size_cutoff + 1;
}

/* ---------------------------------------------------------------------- */

ComputeClusterSizeExt::~ComputeClusterSizeExt() noexcept(true)
{
  if (dist != nullptr) { memory->destroy(dist); }
  if (dist_local != nullptr) { memory->destroy(dist_local); }
  if (counts_global != nullptr) { memory->destroy(counts_global); }
  if (displs != nullptr) { memory->destroy(displs); }
  if (clusters != nullptr) { memory->destroy(clusters); }
  if (ns != nullptr) { memory->destroy(ns); }
  if (gathered != nullptr) { memory->destroy(gathered); }

  // delete cluster_map;
  // delete alloc;
  // delete keeper;
}

/* ---------------------------------------------------------------------- */

void ComputeClusterSizeExt::init()
{
  if ((modify->get_compute_by_style(style).size() > 1) && (comm->me == 0)) {
    error->warning(FLERR, "More than one compute {}", style);
  }

  memory->create(counts_global, comm->nprocs, "counts_global");
  memory->create(displs, comm->nprocs, "displs");

  memory->create(dist_local, size_vector, "compute:cluster/size:dist");
  memory->create(dist, size_vector, "compute:cluster/size:dist");
  vector = dist;

  nloc = static_cast<int>(atom->nlocal * ComputeClusterSizeIm_ALLOC_COEFF);
  cluster_map.reserve(nloc);
  // cluster_map->reserve(nloc);
  // alloc->pool_size_ = nloc;
  memory->create(clusters, nloc, "clusters");
  memory->create(ns, 2 * nloc, "ns");

  natom_loc = static_cast<int>(2 * atom->natoms * ComputeClusterSizeIm_ALLOC_COEFF);
  memory->create(gathered, natom_loc, "gathered");

  initialized_flag = 1;
}

/* ---------------------------------------------------------------------- */

// void ComputeClusterSizeExt::test_allocator() const
// {
//   constexpr ssize_t pool_size = 1024;
//   constexpr size_t len = 3000;

//   // Create a custom allocator instance
//   MemoryKeeper keeper(memory);
//   CustomAllocator<std::pair<const int, int>> allocator(pool_size, memory, &keeper);

//   // Create an unordered_map using the custom allocator
//   std::unordered_map<int, int, std::hash<int>, std::equal_to<int>,
//                       CustomAllocator<std::pair<const int, int>>> map(allocator);

//   // Populate the map
//   for (int i = 0; i < len; ++i) {
//       if (map.count(i) == 0) {
//           map[i] = len-i;
//       }
//   }

//   utils::logmesg(lmp, "Map test passed successfully");
// }

/* ---------------------------------------------------------------------- */

void ComputeClusterSizeExt::compute_vector()
{
  invoked_vector = update->ntimestep;

  auto &cmap = cluster_map;

  if (compute_cluster_atom->invoked_peratom != update->ntimestep) {
    compute_cluster_atom->compute_peratom();
  }

  const double *const cluster_ids = compute_cluster_atom->vector_atom;
  cmap.clear();
  ::memset(counts_global, 0, comm->nprocs * sizeof(int));
  ::memset(displs, 0, comm->nprocs * sizeof(int));
  ::memset(dist_local, 0.0, size_vector * sizeof(double));
  ::memset(dist, 0.0, size_vector * sizeof(double));

  if (atom->nlocal > nloc) {
    nloc = static_cast<int>(atom->nlocal * ComputeClusterSizeIm_ALLOC_COEFF);
    cmap.reserve(nloc);
    // alloc->pool_size_ = nloc;
    memory->grow(clusters, nloc, "clusters");
    memory->grow(ns, 2 * nloc, "ns");
  }

  // Sort atom IDs by cluster IDs
  for (int i = 0; i < atom->nlocal; ++i) {
    if ((atom->mask[i] & groupbit) != 0) {
      int const clid = static_cast<int>(cluster_ids[i]);
      if (cmap.count(clid) == 0) {
        int const idx = cmap.size();
        ns[idx] = clid;
        ns[idx + 1] = 0;
        clusters[idx] = {0, 0, -1, 0, 0, 0};
        cmap[clid] = {ns + 2 * idx, clusters + idx};
      }
      cmap[clid].ptr->atoms[cmap[clid].n[1]++] =
          i;    // possible segfault if actual cluster size exceeds LMP_NUCC_CLUSTER_MAX_SIZE + LMP_NUCC_CLUSTER_MAX_GHOST
    }
  }

  // add ghost atoms
  for (int i = atom->nlocal; i < atom->nmax; ++i) {
    if ((atom->mask[i] & groupbit) != 0) {
      int const clid = static_cast<int>(cluster_ids[i]);
      if (cmap.count(clid) > 0) {
        cmap[clid].ptr->ghost[cmap[clid].ptr->nghost++] =
            i;    // also possible segfault if number of ghost exceeds LMP_NUCC_CLUSTER_MAX_GHOST
      }
    }
  }

  int ncluster_local = 2 * cmap.size();
  ::MPI_Allgather(&ncluster_local, 1, MPI_INT, counts_global, 1, MPI_INT, world);

  int tcon = counts_global[0];
  for (int i = 1; i < comm->nprocs; ++i) {
    tcon += counts_global[i];
    displs[i] = displs[i - 1] + counts_global[i - 1];
  }

  if (tcon > natom_loc) {
    natom_loc = static_cast<int>(tcon * ComputeClusterSizeIm_ALLOC_COEFF);
    memory->grow(gathered, natom_loc, "gathered");
  }

  ::MPI_Allgatherv(ns, ncluster_local, MPI_INT, gathered, counts_global, displs, MPI_INT, world);

  for (int i = 0; i < comm->nprocs; ++i) {
    for (int j = 0; j < counts_global[i] - 1; j += 2) {
      int const k = displs[i] + j;
      if (cmap.count(gathered[k]) > 0) {
        cluster_data &cptr = *(cmap[gathered[k]].ptr);
        cptr.owners[cptr.nowners++] = i;
        cptr.g_size += gathered[k + 1];
        if (gathered[k + 1] > cptr.nhost) {
          cptr.host = i;
          cptr.nhost = gathered[k + 1];
        }
      }
    }
  }

  for (auto &[clid, cptr] : cmap) {
    cluster_data &cpptr = *(cptr.ptr);
    cpptr.l_size = cptr.n[0];
    ::memcpy(cpptr.atoms + cpptr.l_size, cpptr.ghost, cpptr.nghost * sizeof(int));
    if ((cptr.ptr->host < 0) && (cptr.ptr->g_size < size_cutoff)) {
      dist_local[cptr.ptr->g_size] += 1;
    }
  }

  ::MPI_Allreduce(dist_local, dist, size_vector, MPI_DOUBLE, MPI_SUM, world);
}

/* ----------------------------------------------------------------------
   memory usage of maps and dist
------------------------------------------------------------------------- */

double ComputeClusterSizeExt::memory_usage()
{
  double sum = static_cast<unsigned long>(size_vector) * 2 * sizeof(double);
  sum += static_cast<unsigned long>(comm->nprocs) * 2 * sizeof(int);
  sum += static_cast<unsigned long>(nloc) *
      (sizeof(cluster_data) + 2 * sizeof(int) + sizeof(std::pair<int, cluster_ptr>));
  sum += static_cast<unsigned long>(natom_loc) * sizeof(int);
  return sum;
}

/* ---------------------------------------------------------------------- */
