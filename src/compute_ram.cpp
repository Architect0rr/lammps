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

#include "compute_ram.h"
#include "nucc_cspan.hpp"
#include <cmath>
#include <cstddef>

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstring>
#include <fstream>
#include <unordered_map>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeRAMUsage::ComputeRAMUsage(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg)
{
  scalar_flag = 1;
  extscalar = 1;
  local_flag = 1;
  size_local_rows = 0;
  size_local_cols = 0;
  scalar = 0;
  initialized_flag = 1;
}

/* ---------------------------------------------------------------------- */

ComputeRAMUsage::~ComputeRAMUsage() noexcept {}

/* ---------------------------------------------------------------------- */

void ComputeRAMUsage::init() {}

/* ---------------------------------------------------------------------- */

double ComputeRAMUsage::compute_scalar()
{
  invoked_scalar = update->ntimestep;

  compute_local();
  uint64_t usg = 0;
  ::MPI_Allreduce(&local_usage, &usg, 1, MPI_UINT64_T, MPI_SUM, world);
  return scalar = static_cast<double>(usg);
}

/* ---------------------------------------------------------------------- */

uint getCurrentMemoryUsage() {
  std::ifstream file("/proc/self/status");
  std::string line;
  uint64_t memoryUsage = -1;

  if (file.is_open()) {
    while (std::getline(file, line)) {
      if (line.substr(0, 6) == "VmRSS:") {
        int res = std::sscanf(line.c_str(), "VmRSS: %ld kB", &memoryUsage);
        // memoryUsage *= 1024;
        if ((res == 0) || (res == EOF)) { memoryUsage = -1; }
        break;
      }
    }
    file.close();
  }

  return memoryUsage;
}
/* ---------------------------------------------------------------------- */

void ComputeRAMUsage::compute_local()
{
  invoked_local = update->ntimestep;
  local_usage = getCurrentMemoryUsage();
  if (local_usage <= 0) {
    local_usage = 0;
    error->warning(FLERR, "An error occured, while trying to get memory usage from '/proc/self/status'");
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeRAMUsage::memory_usage()
{
  return 1;
}
