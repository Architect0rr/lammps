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

#include "compute_cluster_volume.h"
#include "compute_cluster_size.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstring>
#include <unordered_map>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeClusterVolume::ComputeClusterVolume(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg)
{
  vector_flag = 1;
  size_vector = 0;
  extvector = 0;
  local_flag = 1;
  size_local_rows = 0;
  size_local_cols = 0;

  if (narg < 5) { utils::missing_cmd_args(FLERR, "compute cluster/volume", error); }

  // Parse arguments //

  // Get cluster/size compute
  compute_cluster_size = dynamic_cast<ComputeClusterSize *>(lmp->modify->get_compute_by_id(arg[3]));
  if (compute_cluster_size == nullptr) {
    error->all(FLERR, "compute {}: Cannot find compute with style 'cluster/size' with id: {}",
               style, arg[3]);
  }

  if (::strcmp(arg[4], "rect") == 0) {
    mode = VOLUMEMODE::RECTANGLE;
  } else if (::strcmp(arg[4], "sphere") == 0) {
    mode = VOLUMEMODE::SPHERE;
  } else {
    error->all(FLERR, "Unknown mode for compute {}: {}", style, arg[4]);
  }

  // Get the critical size
  size_cutoff = compute_cluster_size->get_size_cutoff();
  if ((narg >= 6) && (::strcmp(arg[5], "inherit") != 0)) {
    int const t_size_cutoff = utils::inumeric(FLERR, arg[5], true, lmp);
    if (t_size_cutoff < 1) {
      error->all(FLERR, "size_cutoff for compute {} must be greater than 0", style);
    }
    if (t_size_cutoff > size_cutoff) {
      error->all(FLERR,
                 "size_cutoff for compute {} cannot be greater than it of compute cluster/size",
                 style);
    }
  }

  size_local_rows = size_cutoff + 1;
  memory->create(local_kes, size_local_rows + 1, "compute:cluster/ke:local_kes");
  vector_local = local_kes;

  size_vector = size_cutoff + 1;
  memory->create(kes, size_vector + 1, "compute:cluster/ke:kes");
  vector = kes;
}

/* ---------------------------------------------------------------------- */

ComputeClusterVolume::~ComputeClusterVolume() noexcept(true)
{
  if (local_kes != nullptr) { memory->destroy(local_kes); }
  if (kes != nullptr) { memory->destroy(kes); }
}

/* ---------------------------------------------------------------------- */

void ComputeClusterVolume::init()
{
  if ((modify->get_compute_by_style(style).size() > 1) && (comm->me == 0)) {
    error->warning(FLERR, "More than one compute {}", style);
  }
}

/* ---------------------------------------------------------------------- */

void ComputeClusterVolume::compute_vector()
{
  invoked_vector = update->ntimestep;

  compute_local();

  ::memset(kes, 0.0, size_vector * sizeof(double));
  ::MPI_Allreduce(local_kes, kes, size_vector, MPI_DOUBLE, MPI_SUM, world);

  const double *dist = compute_cluster_size->vector;
  for (int i = 0; i < size_vector; ++i) {
    if (dist[i] > 0) { kes[i] /= dist[i]; }
  }
}

/* ---------------------------------------------------------------------- */

double ComputeClusterVolume::occupied_volume(const double **const centers, const int *const list,
                                             const int n, const double r,
                                             const double voxel_size = 0.1)
{
  double min_c[3]{};
  double max_c[3]{};

  // find min and max center coordinates, i.e. bounding box for voxel grid
  min_c[0] = centers[list[0]][0];
  min_c[1] = centers[list[0]][1];
  min_c[2] = centers[list[0]][2];
  max_c[0] = centers[list[0]][0];
  max_c[1] = centers[list[0]][1];
  max_c[2] = centers[list[0]][2];

  for (int i = 0; i < n; ++i) {
    if (centers[list[i]][0] < min_c[0]) { min_c[0] = centers[list[i]][0]; }
    if (centers[list[i]][1] < min_c[1]) { min_c[1] = centers[list[i]][1]; }
    if (centers[list[i]][2] < min_c[2]) { min_c[2] = centers[list[i]][2]; }
    if (centers[list[i]][0] > max_c[0]) { max_c[0] = centers[list[i]][0]; }
    if (centers[list[i]][1] > max_c[1]) { max_c[1] = centers[list[i]][1]; }
    if (centers[list[i]][2] > max_c[2]) { max_c[2] = centers[list[i]][2]; }
  }

  min_c[0] -= r;
  min_c[1] -= r;
  min_c[2] -= r;
  max_c[0] += r;
  max_c[1] += r;
  max_c[2] += r;

  const int nx = static_cast<int>(::ceil((max_c[0] - min_c[0]) / voxel_size));
  const int ny = static_cast<int>(::ceil((max_c[1] - min_c[1]) / voxel_size));
  const int nz = static_cast<int>(::ceil((max_c[2] - min_c[2]) / voxel_size));

  // count occupied cells
  int occupied = 0;
  double const rsq = r * r;
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (int k = 0; k < nz; ++k) {
        for (int l = 0; l < n; ++l) {
          double const dx = min_c[0] + i * voxel_size - centers[list[l]][0];
          double const dy = min_c[1] + j * voxel_size - centers[list[l]][1];
          double const dz = min_c[2] + k * voxel_size - centers[list[l]][2];
          if (dx * dx + dy * dy + dz * dz < rsq) {
            ++occupied;
            break;
          };
        }
      }
    }
  }

  return occupied * voxel_size * voxel_size * voxel_size;
}

double ComputeClusterVolume::occupied_volume2(const double **const centers, const int *const list,
                                              const int n, const double r,
                                              const double voxel_size = 0.1)
{
  double min_c[3]{};
  double max_c[3]{};

  // find min and max center coordinates, i.e. bounding box for voxel grid
  min_c[0] = centers[list[0]][0];
  min_c[1] = centers[list[0]][1];
  min_c[2] = centers[list[0]][2];
  max_c[0] = centers[list[0]][0];
  max_c[1] = centers[list[0]][1];
  max_c[2] = centers[list[0]][2];

  for (int i = 0; i < n; ++i) {
    if (centers[list[i]][0] < min_c[0]) { min_c[0] = centers[list[i]][0]; }
    if (centers[list[i]][1] < min_c[1]) { min_c[1] = centers[list[i]][1]; }
    if (centers[list[i]][2] < min_c[2]) { min_c[2] = centers[list[i]][2]; }
    if (centers[list[i]][0] > max_c[0]) { max_c[0] = centers[list[i]][0]; }
    if (centers[list[i]][1] > max_c[1]) { max_c[1] = centers[list[i]][1]; }
    if (centers[list[i]][2] > max_c[2]) { max_c[2] = centers[list[i]][2]; }
  }

  min_c[0] -= r;
  min_c[1] -= r;
  min_c[2] -= r;
  max_c[0] += r;
  max_c[1] += r;
  max_c[2] += r;

  // count occupied cells
  int occupied = 0;
  double const rsq = r * r;
  double cc[3]{};

  cc[0] = min_c[0];
  while (cc[0] < max_c[0]) {
    cc[1] = min_c[1];
    while (cc[1] < max_c[1]) {
      cc[2] = min_c[2];
      while (cc[2] < max_c[2]) {
        for (int l = 0; l < n; ++l) {
          double const dx = cc[0] - centers[list[l]][0];
          double const dy = cc[1] - centers[list[l]][1];
          double const dz = cc[2] - centers[list[l]][2];
          if (dx * dx + dy * dy + dz * dz < rsq) {
            ++occupied;
            break;
          };
        }
        cc[2] += voxel_size;
      }
      cc[1] += voxel_size;
    }
    cc[0] += voxel_size;
  }

  return occupied * voxel_size * voxel_size * voxel_size;
}

/* ---------------------------------------------------------------------- */

void ComputeClusterVolume::compute_local()
{
  invoked_local = update->ntimestep;

  if (compute_cluster_size->invoked_vector != update->ntimestep) {
    compute_cluster_size->compute_vector();
  }

  ::memset(local_kes, 0.0, size_local_rows * sizeof(double));
  const double *const peratomkes = compute_cluster_size->vector_atom;

  for (const auto &[size, vec] : compute_cluster_size->cIDs_by_size) {
    if (size < size_cutoff) {
      for (const tagint cid : vec) {
        for (const tagint pid : compute_cluster_size->atoms_by_cID[cid]) {
          local_kes[size] += peratomkes[pid];
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeClusterVolume::memory_usage()
{
  return static_cast<double>((size_local_rows + size_vector) * sizeof(double));
}
