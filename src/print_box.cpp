#include "print_box.h"
#include "domain.h"
#include "utils.h"
#include "comm.h"

using namespace LAMMPS_NS;

PrintBox::PrintBox(LAMMPS *lmp) : Command(lmp) {}

void PrintBox::command(int narg, char **arg) {
  if (comm->me != 0) return;

  // Output all members of Domain class, excluding those derived from Pointers
  utils::logmesg(lmp, "box_exist: {} \n", domain->box_exist);
  utils::logmesg(lmp, "dimension: {} \n", domain->dimension);
  utils::logmesg(lmp, "nonperiodic: {} \n", domain->nonperiodic);
  utils::logmesg(lmp, "xperiodic: {} \n", domain->xperiodic);
  utils::logmesg(lmp, "yperiodic: {} \n", domain->yperiodic);
  utils::logmesg(lmp, "zperiodic: {} \n", domain->zperiodic);
  for (int i = 0; i < 3; ++i) {
      utils::logmesg(lmp, "periodicity[{}]: {} \n", i, domain->periodicity[i]);
      for (int j = 0; j < 2; ++j) {
      utils::logmesg(lmp, "boundary[{}][{}]: {} \n", i, j, domain->boundary[i][j]);
      }
  }
  utils::logmesg(lmp, "triclinic: {} \n", domain->triclinic);
  utils::logmesg(lmp, "triclinic_general: {} \n", domain->triclinic_general);
  utils::logmesg(lmp, "xprd: {} \n", domain->xprd);
  utils::logmesg(lmp, "yprd: {} \n", domain->yprd);
  utils::logmesg(lmp, "zprd: {} \n", domain->zprd);
  utils::logmesg(lmp, "xprd_half: {} \n", domain->xprd_half);
  utils::logmesg(lmp, "yprd_half: {} \n", domain->yprd_half);
  utils::logmesg(lmp, "zprd_half: {} \n", domain->zprd_half);
  for (int i = 0; i < 3; ++i) {
      utils::logmesg(lmp, "prd[{}]: {} \n", i, domain->prd[i]);
      utils::logmesg(lmp, "prd_half[{}]: {} \n", i, domain->prd_half[i]);
      utils::logmesg(lmp, "prd_lamda[{}]: {} \n", i, domain->prd_lamda[i]);
      utils::logmesg(lmp, "prd_half_lamda[{}]: {} \n", i, domain->prd_half_lamda[i]);
  }
  for (int i = 0; i < 3; ++i) {
      utils::logmesg(lmp, "boxlo[{}]: {} \n", i, domain->boxlo[i]);
      utils::logmesg(lmp, "boxhi[{}]: {} \n", i, domain->boxhi[i]);
      utils::logmesg(lmp, "boxlo_lamda[{}]: {} \n", i, domain->boxlo_lamda[i]);
      utils::logmesg(lmp, "boxhi_lamda[{}]: {} \n", i, domain->boxhi_lamda[i]);
      utils::logmesg(lmp, "boxlo_bound[{}]: {} \n", i, domain->boxlo_bound[i]);
      utils::logmesg(lmp, "boxhi_bound[{}]: {} \n", i, domain->boxhi_bound[i]);
  }
  for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 3; ++j) {
      utils::logmesg(lmp, "corners[{}][{}]: {} \n", i, j, domain->corners[i][j]);
      }
  }
  utils::logmesg(lmp, "minxlo: {} \n", domain->minxlo);
  utils::logmesg(lmp, "minxhi: {} \n", domain->minxhi);
  utils::logmesg(lmp, "minylo: {} \n", domain->minylo);
  utils::logmesg(lmp, "minyhi: {} \n", domain->minyhi);
  utils::logmesg(lmp, "minzlo: {} \n", domain->minzlo);
  utils::logmesg(lmp, "minzhi: {} \n", domain->minzhi);
  for (int i = 0; i < 3; ++i) {
      utils::logmesg(lmp, "sublo[{}]: {} \n", i, domain->sublo[i]);
      utils::logmesg(lmp, "subhi[{}]: {} \n", i, domain->subhi[i]);
      utils::logmesg(lmp, "sublo_lamda[{}]: {} \n", i, domain->sublo_lamda[i]);
      utils::logmesg(lmp, "subhi_lamda[{}]: {} \n", i, domain->subhi_lamda[i]);
  }
  utils::logmesg(lmp, "xy: {} \n", domain->xy);
  utils::logmesg(lmp, "xz: {} \n", domain->xz);
  utils::logmesg(lmp, "yz: {} \n", domain->yz);
  for (int i = 0; i < 6; ++i) {
      utils::logmesg(lmp, "h[{}]: {} \n", i, domain->h[i]);
      utils::logmesg(lmp, "h_inv[{}]: {} \n", i, domain->h_inv[i]);
      utils::logmesg(lmp, "h_rate[{}]: {} \n", i, domain->h_rate[i]);
  }
  for (int i = 0; i < 3; ++i) {
      utils::logmesg(lmp, "h_ratelo[{}]: {} \n", i, domain->h_ratelo[i]);
  }
  for (int i = 0; i < 3; ++i) {
      utils::logmesg(lmp, "avec[{}]: {} \n", i, domain->avec[i]);
      utils::logmesg(lmp, "bvec[{}]: {} \n", i, domain->bvec[i]);
      utils::logmesg(lmp, "cvec[{}]: {} \n", i, domain->cvec[i]);
  }
  for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
      utils::logmesg(lmp, "rotate_g2r[{}][{}]: {} \n", i, j, domain->rotate_g2r[i][j]);
      utils::logmesg(lmp, "rotate_r2g[{}][{}]: {} \n", i, j, domain->rotate_r2g[i][j]);
      }
  }
  utils::logmesg(lmp, "box_change: {} \n", domain->box_change);
  utils::logmesg(lmp, "box_change_size: {} \n", domain->box_change_size);
  utils::logmesg(lmp, "box_change_shape: {} \n", domain->box_change_shape);
  utils::logmesg(lmp, "box_change_domain: {} \n", domain->box_change_domain);
  utils::logmesg(lmp, "deform_flag: {} \n", domain->deform_flag);
  utils::logmesg(lmp, "deform_vremap: {} \n", domain->deform_vremap);
  utils::logmesg(lmp, "deform_groupbit: {} \n", domain->deform_groupbit);
  utils::logmesg(lmp, "copymode: {} \n", domain->copymode);

  double prd_half_lam[3];
  domain->x2lamda(domain->prd_half, prd_half_lam);
  double prd_half[3];
  domain->lamda2x(prd_half_lam, prd_half);
  utils::logmesg(lmp, "x_h: {} -x2l-> {} -l2x-> {} \n", domain->xprd_half, prd_half_lam[0], prd_half[0]);
  utils::logmesg(lmp, "y_h: {} -x2l-> {} -l2x-> {} \n", domain->yprd_half, prd_half_lam[1], prd_half[1]);
  utils::logmesg(lmp, "z_h: {} -x2l-> {} -l2x-> {} \n", domain->zprd_half, prd_half_lam[2], prd_half[2]);
}