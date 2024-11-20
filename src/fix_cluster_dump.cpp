/*
 ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "fix_cluster_dump.h"

#include "comm.h"
#include "error.h"
#include "fix.h"
#include "fmt/core.h"
#include "modify.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixClusterDump::FixClusterDump(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{

  restart_pbc = 1;

  nevery = 1;

  if (narg < 14) { utils::missing_cmd_args(FLERR, "cluster/dump", error); }

  // Parse arguments //

  // Get the nevery
  nevery = utils::inumeric(FLERR, arg[3], true, lmp);
  if (nevery < 1) { error->all(FLERR, "nevery for cluster/dump nust be greater than 0"); }

  // Get the critical size
  size_cutoff = utils::inumeric(FLERR, arg[4], true, lmp);
  if (size_cutoff < 1) { error->all(FLERR, "size_cutoff for cluster/dump must be greater than 0"); }

  // Get cluster/size compute
  compute_cluster_size = lmp->modify->get_compute_by_id(arg[5]);
  if (compute_cluster_size == nullptr) {
    error->all(FLERR, "cluster/dump: Cannot find compute of style 'cluster/size' with id: {}",
               arg[5]);
  }

  // Get cluster/temp compute
  compute_cluster_temp = lmp->modify->get_compute_by_id(arg[6]);
  if (compute_cluster_temp == nullptr) {
    error->all(FLERR, "cluster/dump: Cannot find compute of style 'cluster/temp' with id: {}",
               arg[6]);
  }

  // Get supersaturation/mono compute
  compute_supersaturation_mono = lmp->modify->get_compute_by_id(arg[7]);
  if (compute_supersaturation_mono == nullptr) {
    error->all(FLERR,
               "cluster/dump: Cannot find compute of style 'supersaturation/mono' with id: {}",
               arg[7]);
  }

  // Get supersaturation/density compute
  compute_supersaturation_density = lmp->modify->get_compute_by_id(arg[8]);
  if (compute_supersaturation_density == nullptr) {
    error->all(FLERR,
               "cluster/dump: Cannot find compute of style 'supersaturation/density' with id: {}",
               arg[8]);
  }

  // Get cluster/ke compute
  compute_cluster_ke = lmp->modify->get_compute_by_id(arg[9]);
  if (compute_cluster_ke == nullptr) {
    error->all(FLERR, "cluster/dump: Cannot find compute of style 'cluster/temp' with id: {}",
               arg[9]);
  }

  // // Get ke/mono compute
  // fix_kedff = dynamic_cast<FixKedff *>(lmp->modify->get_fix_by_id(arg[14]));
  // if (fix_kedff == nullptr) {
  //   error->all(FLERR, "cluster/dump: Cannot find fix of style 'kedff' with id: {}", arg[14]);
  // }

  if (comm->me == 0) {
    cldist = ::fopen(arg[10], "a");
    if (cldist == nullptr) {
      error->one(FLERR, "Cannot open file {}: {}", arg[10], utils::getsyserror());
    }

    cltemp = ::fopen(arg[11], "a");
    if (cltemp == nullptr) {
      error->one(FLERR, "Cannot open file {}: {}", arg[11], utils::getsyserror());
    }

    clke = ::fopen(arg[12], "a");
    if (clke == nullptr) {
      error->one(FLERR, "Cannot open file {}: {}", arg[12], utils::getsyserror());
    }

    scalars = ::fopen(arg[13], "a");
    if (scalars == nullptr) {
      error->one(FLERR, "Cannot open file {}: {}", arg[13], utils::getsyserror());
    }
    fmt::print(scalars, "ntimestep,T,Srho,S1\n");
    ::fflush(scalars);
  }

  // Get temp compute
  auto computes = lmp->modify->get_compute_by_style("temp");
  if (computes.empty()) { error->all(FLERR, "cluster/dump: Cannot find compute of style 'temp'"); }
  compute_temp = computes[0];
}

/* ---------------------------------------------------------------------- */

FixClusterDump::~FixClusterDump() noexcept(true)
{
  if (comm->me == 0) {
    if (cldist != nullptr) {
      ::fflush(cldist);
      ::fclose(cldist);
    }
    if (cltemp != nullptr) {
      ::fflush(cltemp);
      ::fclose(cltemp);
    }
    if (clke != nullptr) {
      ::fflush(clke);
      ::fclose(clke);
    }
    if (scalars != nullptr) {
      ::fflush(scalars);
      ::fclose(scalars);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixClusterDump::init()
{
  if ((modify->get_fix_by_style(style).size() > 1) && (comm->me == 0)) {
    error->warning(FLERR, "More than one fix {}", style);
  }
}

/* ---------------------------------------------------------------------- */

int FixClusterDump::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixClusterDump::end_of_step()
{
  if (compute_temp->invoked_scalar != update->ntimestep) { compute_temp->compute_scalar(); }

  if (compute_cluster_size->invoked_vector != update->ntimestep) {
    compute_cluster_size->compute_vector();
  }

  if (compute_cluster_temp->invoked_vector != update->ntimestep) {
    compute_cluster_temp->compute_vector();
  }

  if (compute_cluster_ke->invoked_vector != update->ntimestep) {
    compute_cluster_ke->compute_vector();
  }

  // if (fix_kedff->invoked_endofstep != update->ntimestep) { fix_kedff->end_of_step(); }

  if (compute_supersaturation_density->invoked_scalar != update->ntimestep) {
    compute_supersaturation_density->compute_scalar();
  }

  if (compute_supersaturation_mono->invoked_scalar != update->ntimestep) {
    compute_supersaturation_mono->compute_scalar();
  }

  const bigint dist_size = compute_cluster_size->size_vector - 1;
  const bigint write_cutoff = MIN(size_cutoff, dist_size);

  const double *const dist = compute_cluster_size->vector;
  const double *const temp = compute_cluster_temp->vector;
  const double *const ke = compute_cluster_ke->vector;

  if (comm->me == 0) {

    fmt::print(cldist, "{},", update->ntimestep);
    for (bigint i = 1; i < write_cutoff; ++i) {
      fmt::print(cldist, "{},", static_cast<bigint>(dist[i]));
    }
    fmt::print(cldist, "{}\n", static_cast<bigint>(dist[write_cutoff]));
    ::fflush(cldist);

    fmt::print(cltemp, "{},", update->ntimestep);
    for (bigint i = 1; i < write_cutoff; ++i) { fmt::print(cltemp, "{:.5f},", temp[i]); }
    fmt::print(cltemp, "{:.5f}\n", temp[write_cutoff]);
    ::fflush(cltemp);

    fmt::print(clke, "{},", update->ntimestep);
    for (bigint i = 1; i < write_cutoff; ++i) { fmt::print(clke, "{:.5f},", ke[i]); }
    fmt::print(clke, "{:.5f}\n", ke[write_cutoff]);
    ::fflush(clke);

    fmt::print(scalars, "{},{:.5f},{:.5f},{:.5f}\n", update->ntimestep, compute_temp->scalar,
               compute_supersaturation_density->scalar, compute_supersaturation_mono->scalar);
    //fix_kedff->engs_global[0], fix_kedff->engs_global[1]);
    ::fflush(scalars);
  }

}    // void FixClusterCrush::end_of_step()
