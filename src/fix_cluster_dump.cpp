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
  next_step = 0;
  nevery = 1;

  if (narg < 9) utils::missing_cmd_args(FLERR, "cluster/dump", error);

  // Parse arguments //

  // Get the nevery
  nevery = utils::numeric(FLERR, arg[3], true, lmp);
  if (nevery < 1) error->all(FLERR, "nevery for cluster/dump nust be greater than 0");

  // Get the critical size
  size_cutoff = utils::numeric(FLERR, arg[4], true, lmp);
  if (size_cutoff < 1) error->all(FLERR, "size_cutoff for cluster/dump must be greater than 0");

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

  if (comm->me == 0) {
    cldist = fopen(arg[9], "a");
    if (cldist == nullptr)
      error->one(FLERR, "Cannot open file {}: {}", arg[9], utils::getsyserror());

    cltemp = fopen(arg[10], "a");
    if (cltemp == nullptr)
      error->one(FLERR, "Cannot open file {}: {}", arg[10], utils::getsyserror());

    scalars = fopen(arg[11], "a");
    if (scalars == nullptr)
      error->one(FLERR, "Cannot open file {}: {}", arg[11], utils::getsyserror());
    fmt::print(scalars, "ntimestep,T,Srho,S1\n");
    fflush(scalars);
  }

  // Get temp compute
  auto computes = lmp->modify->get_compute_by_style("temp");
  if (computes.size() < 1) {
    error->all(FLERR, "cluster/dump: Cannot find compute of style 'temp'");
  }
  compute_temp = computes[0];

  next_step = update->ntimestep - (update->ntimestep % nevery);
}

/* ---------------------------------------------------------------------- */

FixClusterDump::~FixClusterDump()
{
  if (comm->me == 0) {
    if (cldist != nullptr) {
      fflush(cldist);
      fclose(cldist);
    }
    if (cltemp != nullptr) {
      fflush(cltemp);
      fclose(cltemp);
    }
    if (scalars != nullptr) {
      fflush(scalars);
      fclose(scalars);
    }
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

void FixClusterDump::init() {}

/* ---------------------------------------------------------------------- */

void FixClusterDump::end_of_step()
{
  if (update->ntimestep < next_step) return;
  next_step = update->ntimestep + nevery;

  if (compute_temp->invoked_scalar != update->ntimestep) { compute_temp->compute_scalar(); }

  if (compute_cluster_size->invoked_vector != update->ntimestep) {
    compute_cluster_size->compute_vector();
  }

  if (compute_cluster_temp->invoked_vector != update->ntimestep) {
    compute_cluster_temp->compute_vector();
  }

  if (compute_supersaturation_density->invoked_scalar != update->ntimestep) {
    compute_supersaturation_density->compute_scalar();
  }

  if (compute_supersaturation_mono->invoked_scalar != update->ntimestep) {
    compute_supersaturation_mono->compute_scalar();
  }

  bigint dist_size = compute_cluster_size->size_vector - 1;
  bigint write_cutoff = (size_cutoff < dist_size ? size_cutoff : dist_size);

  double *dist = compute_cluster_size->vector;
  double *temp = compute_cluster_temp->vector;

  if (comm->me == 0) {
    fprintf(cldist, "%d,", update->ntimestep);
    for (bigint i = 1; i < write_cutoff; ++i) {
      fprintf(cldist, "%d,", static_cast<bigint>(dist[i]));
    }
    fprintf(cldist, "%d\n", static_cast<bigint>(dist[write_cutoff]));

    fprintf(cltemp, "%d,", update->ntimestep);
    for (bigint i = 1; i < write_cutoff; ++i) { fprintf(cltemp, "%.5f,", temp[i]); }
    fprintf(cltemp, "%.5f\n", temp[write_cutoff]);

    fmt::print(scalars, "{},{},{},{}\n", update->ntimestep, compute_temp->scalar,
               compute_supersaturation_density->scalar, compute_supersaturation_mono->scalar);
    fflush(cldist);
    fflush(cltemp);
    fflush(scalars);
  }

}    // void FixClusterCrush::end_of_step()