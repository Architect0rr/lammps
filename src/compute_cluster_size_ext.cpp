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
#include "nucc_cspan.hpp"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstddef>
#include <cstring>
#include <utility>

using namespace LAMMPS_NS;
using namespace NUCC;

/* ---------------------------------------------------------------------- */

ComputeClusterSizeExt::ComputeClusterSizeExt(LAMMPS* lmp, int narg, char** arg) :
    Compute(lmp, narg, arg), nloc(0), nc_global(0), natom_loc(0), nloc_peratom(0)
{
  vector_flag = 1;
  extvector = 0;
  size_vector = 0;
  size_vector_variable = 1;

  peratom_flag = 1;
  size_peratom_cols = 0;

  if (comm->nprocs > LMP_NUCC_CLUSTER_MAX_OWNERS) {
    error->all(FLERR, "Number of processor exceeds MAX_OWNER limit. Recompile with higher MAX_OWNER limit.");
  }

  if (narg < 4) { utils::missing_cmd_args(FLERR, "compute cluster/size", error); }

  // Parse arguments //

  // Get cluster/atom compute
  compute_cluster_atom = lmp->modify->get_compute_by_id(arg[3]);
  if (compute_cluster_atom == nullptr) { error->all(FLERR, "{}: Cannot find compute with style 'cluster/atom' with given id: {}", style, arg[3]); }

  // Get the critical size
  size_cutoff = utils::inumeric(FLERR, arg[4], true, lmp);
  if (size_cutoff < 1) { error->all(FLERR, "size_cutoff for compute cluster/size must be greater than 0"); }

  keeper1 = new MemoryKeeper(memory);
  cluster_map_allocator = new MapAlloc_t<int, int>(keeper1);
  cluster_map = new Map_t<int, int>(*cluster_map_allocator);

  keeper2 = new MemoryKeeper(memory);
  alloc_map_vec1 = new MapAlloc_t<int, Vec_t<int>>(keeper2);
  cIDs_by_size = new Map_t<int, Vec_t<int>>(*alloc_map_vec1);

  keeper3 = new MemoryKeeper(memory);
  alloc_map_vec2 = new MapAlloc_t<int, Vec_t<int>>(keeper3);
  cIDs_by_size_all = new Map_t<int, Vec_t<int>>(*alloc_map_vec2);

  size_vector = size_cutoff + 1;
}

/* ---------------------------------------------------------------------- */

ComputeClusterSizeExt::~ComputeClusterSizeExt() noexcept(true)
{
  dist.destroy(memory);
  dist_local.destroy(memory);
  counts_global.destroy(memory);
  displs.destroy(memory);
  clusters.destroy(memory);
  ns.destroy(memory);
  gathered.destroy(memory);
  monomers.destroy(memory);

  delete keeper1;
  delete cluster_map_allocator;
  delete cluster_map;

  delete keeper2;
  delete alloc_map_vec1;
  delete cIDs_by_size;

  delete keeper3;
  delete alloc_map_vec2;
  delete cIDs_by_size_all;
}

/* ---------------------------------------------------------------------- */

void ComputeClusterSizeExt::init()
{
  if ((modify->get_compute_by_style(style).size() > 1) && (comm->me == 0)) { error->warning(FLERR, "More than one compute {}", style); }

  counts_global.create(memory, comm->nprocs, "counts_global");
  displs.create(memory, comm->nprocs, "displs");

  dist_local.create(memory, size_vector, "compute:cluster/size:dist");
  dist.create(memory, size_vector, "compute:cluster/size:dist");
  vector = dist.data();

  nloc = static_cast<int>(atom->nlocal * LMP_NUCC_ALLOC_COEFF);
  keeper1->pool_size<MapMember_t<int, int>>(nloc);
  keeper2->pool_size<MapMember_t<int, Vec_t<int>>>(nloc);
  keeper3->pool_size<MapMember_t<int, Vec_t<int>>>(nloc);

  cluster_map->reserve(nloc);
  cIDs_by_size->reserve(size_cutoff);
  cIDs_by_size_all->reserve(size_cutoff);

  clusters.create(memory, nloc, "clusters");
  ns.create(memory, 2 * nloc, "ns");
  monomers.create(memory, nloc, "monomers");

  natom_loc = static_cast<bigint>(static_cast<long double>(2 * atom->natoms) * LMP_NUCC_ALLOC_COEFF);
  gathered.create(memory, natom_loc, "gathered");

  nloc_peratom = static_cast<int>(atom->nlocal * LMP_NUCC_ALLOC_COEFF);
  peratom_size.create(memory, nloc_peratom, "cluster/size/ext:peratom");

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

  auto& cmap = *cluster_map;
  auto& cbs = *cIDs_by_size;
  auto& cbs_all = *cIDs_by_size;
  // auto &cmap = cluster_map;

  if (compute_cluster_atom->invoked_peratom != update->ntimestep) { compute_cluster_atom->compute_peratom(); }

  const double* const cluster_ids = compute_cluster_atom->vector_atom;
  cmap.clear();
  cbs.clear();
  cbs_all.clear();
  counts_global.reset();
  displs.reset();
  dist_local.reset();
  dist.reset();

  if (atom->nlocal > nloc) {
    nloc = static_cast<int>(atom->nlocal * LMP_NUCC_ALLOC_COEFF);
    cmap.reserve(nloc);

    clusters.grow(memory, nloc, "clusters");
    ns.grow(memory, 2 * nloc, "ns");
    monomers.grow(memory, nloc, "monomers");
  }

  // Sort atom IDs by cluster IDs
  for (int i = 0; i < atom->nlocal; ++i) {
    if ((atom->mask[i] & groupbit) != 0) {
      const auto clid = static_cast<int>(cluster_ids[i]);
      // if (cluster_ids[i] - clid >= 1) { utils::logmesg(lmp, "Clid mismatch------"); }
      if (cmap.count(clid) == 0) {
        const int clidx = cmap.size();
        cmap[clid] = clidx;
        ns[2 * clidx] = clid;
        ns[2 * clidx + 1] = 0;
        clusters[clidx] = cluster_data(clid);
      }
      // possible segfault if actual cluster size exceeds LMP_NUCC_CLUSTER_MAX_SIZE + LMP_NUCC_CLUSTER_MAX_GHOST
      const int clidx = cmap[clid];
      clusters[clidx].atoms<false>()[ns[2 * clidx + 1]++] = i;
    }
  }

  // add ghost atoms
  for (int i = atom->nlocal; i < atom->nmax; ++i) {
    if ((atom->mask[i] & groupbit) != 0) {
      const auto clid = static_cast<int>(cluster_ids[i]);
      if (cmap.count(clid) > 0) {
        cluster_data& clstr = clusters[cmap[clid]];
        // also possible segfault if number of ghost exceeds LMP_NUCC_CLUSTER_MAX_GHOST
        clstr.ghost_initial()[clstr.nghost++] = i;
      }
    }
  }

  // communicate about number of unique clusters
  const int ncluster_local = 2 * cmap.size();
  ::MPI_Allgather(&ncluster_local, 1, MPI_INT, counts_global.data(), 1, MPI_INT, world);

  int tcon = counts_global[0];
  for (int i = 1; i < comm->nprocs; ++i) {
    tcon += counts_global[i];
    displs[i] = displs[i - 1] + counts_global[i - 1];
  }

  if (tcon > natom_loc) {
    natom_loc = static_cast<int>(tcon * LMP_NUCC_ALLOC_COEFF);
    gathered.grow(memory, natom_loc, "gathered");
  }

  // communicate about local cluster sizes
  ::MPI_Allgatherv(ns.data(), ncluster_local, MPI_INT, gathered.data(), counts_global.data(), displs.data(), MPI_INT, world);

  // fill local data
  for (int i = 0; i < comm->nprocs; ++i) {
    for (int j = 0; j < counts_global[i] - 1; j += 2) {
      int const k = displs[i] + j;
      if (cmap.count(gathered[k]) > 0) {
        cluster_data& clstr = clusters[cmap[gathered[k]]];
        if (i != comm->me) { clstr.owners<false>()[clstr.nowners++] = i; }
        clstr.g_size += gathered[k + 1];
        if (gathered[k + 1] > clstr.nhost) {
          clstr.host = i;
          // if ((clstr.host < 0) || (clstr.host >= comm->nprocs)) { utils::logmesg(lmp, "CS(1): Invalid host: {}", clstr.host); }
          clstr.nhost = gathered[k + 1];
        }
      }
    }
  }

  if (nloc_peratom < atom->nlocal) {
    nloc_peratom = atom->nlocal;
    peratom_size.grow(memory, nloc_peratom, "cluster_size:peratom");
    peratom_size.reset();
    vector_atom = peratom_size.data();
  }

  // adjust local data and fill local size distribution
  nonexclusive = 0;
  nmono = 0;
  for (const auto& [clid, clidx] : cmap) {
    cluster_data& clstr = clusters[clidx];
    clstr.l_size = ns[2 * clidx + 1];
    clstr.rearrange();
    for (int i = 0; i < clstr.l_size; ++i) { peratom_size[clstr.atoms()[i]] = clstr.g_size; }
    if ((clstr.g_size < size_cutoff) && (clstr.g_size > 1)) { cbs_all[clstr.g_size].push_back(clidx); }
    if (clstr.host == comm->me) {
      if (clstr.g_size < size_cutoff) { dist_local[clstr.g_size] += 1; }
      if (clstr.g_size == 1) {
        monomers[nmono++] = clidx;
      } else {
        cbs[clstr.g_size].push_back(clidx);
      }
    }
    if (clstr.nowners > 0) { ++nonexclusive; }
    // if ((clstr.host < 0) || (clstr.host >= comm->nprocs)) { utils::logmesg(lmp, "CS(2): Invalid host: {}", clstr.host); }
    // utils::logmesg(lmp, "Cluster: {}, lsize: {}, nowners: {}, owners:", clid, clstr.l_size, clstr.nowners);
    // for (int i = 0; i < clstr.nowners; ++i) { utils::logmesg(lmp, " {:3}", clstr.owners[i]); }
    // utils::logmesg(lmp, "\n");
  }

  // for (auto &[clid, clidx] : cmap) {
  //   if (clid != clusters[clidx].clid) { utils::logmesg(lmp, "Clid mismatch\n"); }
  //   ns[2 * clidx] = clid;
  //   ns[2 * clidx + 1] = clusters[clidx].host;
  // }
  // ::MPI_Allgatherv(ns, ncluster_local, MPI_INT, gathered, counts_global, displs, MPI_INT, world);

  // int mesgs = 0;
  // for (int i = 0; i < comm->nprocs; ++i) {
  //   for (int j = 0; j < counts_global[i] - 1; j += 2) {
  //     int const k = displs[i] + j;
  //     if (cmap.count(gathered[k]) > 0) {
  //       if (clusters[cmap[gathered[k]]].clid != gathered[k]) {
  //         utils::logmesg(lmp, "{}: Clid mismatch: {}/{}\n", comm->me, clusters[cmap[gathered[k]]].clid, gathered[k]);
  //       }
  //       if (clusters[cmap[gathered[k]]].host != gathered[k+1]) {
  //         cluster_data &clstr = clusters[cmap[gathered[k]]];
  //         utils::logmesg(lmp, "{}: cluster: {}/{}, host: {}, but proc {} have host: {}\n", comm->me, clstr.clid, gathered[k], clstr.host, i, gathered[k+1]);
  //       }
  //       ++mesgs;
  //     }
  //     if (mesgs > 5) break;
  //   }
  //   if (mesgs > 5) break;
  // }

  ::MPI_Allreduce(dist_local.data(), dist.data(), size_vector, MPI_DOUBLE, MPI_SUM, world);
}

/* ---------------------------------------------------------------------- */

void ComputeClusterSizeExt::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  if (invoked_vector != update->ntimestep) { compute_vector(); }
}

/* ----------------------------------------------------------------------
   memory usage of maps and dist
------------------------------------------------------------------------- */

double ComputeClusterSizeExt::memory_usage()
{
  std::size_t sum = dist.memory_usage() + dist_local.memory_usage();
  sum += counts_global.memory_usage() + displs.memory_usage();
  sum += clusters.memory_usage();
  sum += ns.memory_usage() + gathered.memory_usage();
  sum += monomers.memory_usage();
  return static_cast<double>(sum);
}

/* ---------------------------------------------------------------------- */