// clang-format off
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

#include "compute.h"
#include "fix_regen.h"

#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "group.h"
#include "input.h"
#include "lattice.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "molecule.h"
#include "random_park.h"
#include "region.h"
#include "update.h"
#include "variable.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{ATOM,MOLECULE};
enum{DIST_UNIFORM,DIST_GAUSSIAN};

static constexpr double EPSILON = 1.0e6;

/* ---------------------------------------------------------------------- */

FixRegen::FixRegen(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), idregion(nullptr), idrigid(nullptr),
  idshake(nullptr), onemols(nullptr), molfrac(nullptr), coords(nullptr), imageflags(nullptr),
  fixrigid(nullptr), fixshake(nullptr), random(nullptr)
{
  if (narg < 7) { error->all(FLERR,"Illegal fix deposit command");}

  scalar_flag = 1;
  extscalar = 0;
  restart_global = 1;
  time_depend = 1;

  // required args

  ninsert = utils::inumeric(FLERR, arg[3], false, lmp);
  ntype = utils::expand_type_int(FLERR, arg[4], Atom::ATOM, lmp);
  nfreq = utils::inumeric(FLERR, arg[5], false, lmp);
  seed = utils::inumeric(FLERR, arg[6], false, lmp);
  at_once = utils::inumeric(FLERR, arg[7], false, lmp);

  if (seed <= 0) { error->all(FLERR,"Illegal fix deposit command");}

  // read options from end of input line

  options(narg-8,&arg[8]);

  // error check on type

  if (mode == ATOM && (ntype <= 0 || ntype > atom->ntypes)) {
    error->all(FLERR,"Invalid atom type in fix deposit command");}

  // error checks on region and its extent being inside simulation box

  if (iregion == nullptr) { error->all(FLERR,"Must specify a region in fix deposit");}
  if (iregion->bboxflag == 0) {
    error->all(FLERR,"Fix deposit region does not support a bounding box");}
  if (iregion->dynamic_check() != 0) {
    error->all(FLERR,"Fix deposit region cannot be dynamic");}

  xlo = iregion->extent_xlo;
  xhi = iregion->extent_xhi;
  ylo = iregion->extent_ylo;
  yhi = iregion->extent_yhi;
  zlo = iregion->extent_zlo;
  zhi = iregion->extent_zhi;

  if (domain->triclinic == 0) {
    if (xlo < domain->boxlo[0] || xhi > domain->boxhi[0] ||
        ylo < domain->boxlo[1] || yhi > domain->boxhi[1] ||
        zlo < domain->boxlo[2] || zhi > domain->boxhi[2]) {
      error->all(FLERR,"Deposition region extends outside simulation box");}
  } else {
    if (xlo < domain->boxlo_bound[0] || xhi > domain->boxhi_bound[0] ||
        ylo < domain->boxlo_bound[1] || yhi > domain->boxhi_bound[1] ||
        zlo < domain->boxlo_bound[2] || zhi > domain->boxhi_bound[2]) {
      error->all(FLERR,"Deposition region extends outside simulation box");}
  }

  // error check and further setup for mode = MOLECULE

  if (atom->tag_enable == 0) {
    error->all(FLERR,"Cannot use fix_deposit unless atoms have IDs");}

  if (mode == MOLECULE) {
    for (int i = 0; i < nmol; i++) {
      if (onemols[i]->xflag == 0) {
        error->all(FLERR,"Fix deposit molecule must have coordinates");}
      if (onemols[i]->typeflag == 0) {
        error->all(FLERR,"Fix deposit molecule must have atom types");}
      if (ntype+onemols[i]->ntypes <= 0 ||
          ntype+onemols[i]->ntypes > atom->ntypes) {
        error->all(FLERR,"Invalid atom type in fix deposit mol command");}

      if (atom->molecular == Atom::TEMPLATE && onemols != atom->avec->onemols) {
        error->all(FLERR,"Fix deposit molecule template ID must be same "
                   "as atom_style template ID");}
      onemols[i]->check_attributes();

      // fix deposit uses geoemetric center of molecule for insertion

      onemols[i]->compute_center();
    }
  }

  if ((rigidflag != 0) && mode == ATOM) {
    error->all(FLERR,"Cannot use fix deposit rigid and not molecule");}
  if ((shakeflag != 0) && mode == ATOM) {
    error->all(FLERR,"Cannot use fix deposit shake and not molecule");}
  if ((rigidflag != 0) && (shakeflag != 0)) {
    error->all(FLERR,"Cannot use fix deposit rigid and shake");}

  // setup of coords and imageflags array

  if (mode == ATOM) { natom_max = 1;
  } else {
    natom_max = 0;
    for (int i = 0; i < nmol; i++) {
      natom_max = MAX(natom_max,onemols[i]->natoms);}
  }
  memory->create(coords,natom_max,3,"deposit:coords");
  memory->create(imageflags,natom_max,"deposit:imageflags");

  // setup scaling

  double xscale{};
  double yscale{};
  double zscale{};
  if (scaleflag != 0) {
    xscale = domain->lattice->xlattice;
    yscale = domain->lattice->ylattice;
    zscale = domain->lattice->zlattice;
  }
  else { xscale = yscale = zscale = 1.0;}

  // apply scaling to all input parameters with dist/vel units

  if (domain->dimension == 2) {
    lo *= yscale;
    hi *= yscale;
    rate *= yscale;
  } else {
    lo *= zscale;
    hi *= zscale;
    rate *= zscale;
  }
  deltasq *= xscale*xscale;
  nearsq *= xscale*xscale;
  vxlo *= xscale;
  vxhi *= xscale;
  vylo *= yscale;
  vyhi *= yscale;
  vzlo *= zscale;
  vzhi *= zscale;
  xmid *= xscale;
  ymid *= yscale;
  zmid *= zscale;
  sigma *= xscale; // same as in region sphere
  tx *= xscale;
  ty *= yscale;
  tz *= zscale;

  // find current max atom and molecule IDs if necessary

  if (idnext != 0) { find_maxid();}

  // random number generator, same for all procs
  // warm up the generator 30x to avoid correlations in first-particle
  // positions if runs are repeated with consecutive seeds

  random = new RanPark(lmp,seed);
  for (int ii=0; ii < 30; ii++) { random->uniform();}

  // set up reneighboring

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;
  nfirst = next_reneighbor-nfreq;
  ninserted = 0;}

/* ---------------------------------------------------------------------- */

FixRegen::~FixRegen()
{
  delete random;
  delete [] molfrac;
  delete [] idrigid;
  delete [] idshake;
  delete [] idregion;
  delete [] vstr;
  delete [] xstr;
  delete [] ystr;
  delete [] zstr;
  memory->destroy(coords);
  memory->destroy(imageflags);}

/* ---------------------------------------------------------------------- */

int FixRegen::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;}

/* ---------------------------------------------------------------------- */

void FixRegen::init()
{
  warnflag = 1;

  // set index and check validity of region

  iregion = domain->get_region_by_id(idregion);
  if (iregion == nullptr) { error->all(FLERR,"Region ID {} for fix deposit does not exist", idregion);}

  // if rigidflag defined, check for rigid/small fix
  // its molecule template must be same as this one

  fixrigid = nullptr;
  if (rigidflag != 0) {
    fixrigid = modify->get_fix_by_id(idrigid);
    if (fixrigid == nullptr) { error->all(FLERR,"Fix deposit rigid fix ID {} does not exist", idrigid);}
    int tmp = 0;
    if (onemols != static_cast<Molecule **>(fixrigid->extract("onemol",tmp))) {
      error->all(FLERR, "Fix deposit and rigid fix are not using the same molecule template ID");}
  }

  // if shakeflag defined, check for SHAKE fix
  // its molecule template must be same as this one

  fixshake = nullptr;
  if (shakeflag != 0) {
    fixshake = modify->get_fix_by_id(idshake);
    if (fixshake == nullptr) { error->all(FLERR,"Fix deposit shake fix ID {} does not exist", idshake);}
    int tmp = 0;
    if (onemols != static_cast<Molecule **>(fixshake->extract("onemol",tmp))) {
      error->all(FLERR,"Fix deposit and fix shake are not using the same molecule template ID");}
  }

  // for finite size spherical particles:
  // warn if near < 2 * maxrad of existing and inserted particles
  //   since may lead to overlaps
  // if inserted molecule does not define diameters,
  //   use AtomVecSphere::create_atom() default radius = 0.5

  if (atom->radius_flag != 0) {
    double *radius = atom->radius;
    int const nlocal = atom->nlocal;

    double maxrad = 0.0;
    for (int i = 0; i < nlocal; i++) {
      maxrad = MAX(maxrad,radius[i]);}

    double maxradall = NAN;
    MPI_Allreduce(&maxrad,&maxradall,1,MPI_DOUBLE,MPI_MAX,world);

    double maxradinsert = 0.0;
    if (mode == MOLECULE) {
      for (int i = 0; i < nmol; i++) {
        if (onemols[i]->radiusflag != 0) {
          maxradinsert = MAX(maxradinsert,onemols[i]->maxradius);
        } else { maxradinsert = MAX(maxradinsert,0.5);}
      }
    } else { maxradinsert = 0.5;}

    double separation = MAX(2.0*maxradinsert,maxradall+maxradinsert);
    if (sqrt(nearsq) < separation && comm->me == 0) {
      error->warning(FLERR,"Fix deposit near setting < possible overlap separation {}",separation);}
  }

  if (tempflag != 0) {
    if (atom->mass_setflag[ntype] == 0) { error->all(FLERR, "Fix regen: mass is not set for type {}", ntype); }
    vsigma = ::sqrt(temperature / atom->mass[ntype]);
  }
}

/* ---------------------------------------------------------------------- */

void FixRegen::setup_pre_exchange()
{
  if (ninserted < ninsert) { next_reneighbor = nfirst + ((update->ntimestep - nfirst)/nfreq)*nfreq + nfreq;
  } else { next_reneighbor = 0;}
}

/* ----------------------------------------------------------------------
   perform particle insertion
------------------------------------------------------------------------- */

void FixRegen::pre_exchange()
{
  int i{};
  int m{};
  int n{};
  int nlocalprev{};
  int imol{};
  int natom{};
  int flag{};
  int flagall{};
  double coord[3];
  double lamda[3];
  double delx{};
  double dely{};
  double delz{};
  double rsq{};
  double r[3];
  double vnew[3];
  double rotmat[3][3];
  double quat[4];
  double *newcoord = nullptr;

  // just return if should not be called on this timestep

  if (next_reneighbor != update->ntimestep) { return;}

  // clear ghost count (and atom map) and any ghost bonus data
  //   internal to AtomVec
  // same logic as beginning of Comm::exchange()
  // do it now b/c inserting atoms will overwrite ghost atoms

  if (atom->map_style != Atom::MAP_NONE) { atom->map_clear();}
  atom->nghost = 0;
  atom->avec->clear_bonus();

  // compute current offset = bottom of insertion volume

  double offset = 0.0;
  if (rateflag != 0) { offset = (update->ntimestep - nfirst) * update->dt * rate;}

  double *sublo = nullptr;
  double *subhi = nullptr;
  if (domain->triclinic == 0) {
    sublo = domain->sublo;
    subhi = domain->subhi;
  } else {
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  }

  for (int added = 0; added < at_once; ++added) {
    // find current max atom and molecule IDs if necessary

    if (idnext == 0) { find_maxid();}

    // attempt an insertion until successful

    int const dimension = domain->dimension;

    int success = 0;
    int attempt = 0;
    while (attempt < maxattempt) {
      attempt++;

      // choose random position for new particle within region
      if (distflag == DIST_UNIFORM) {
        do {
          coord[0] = xlo + random->uniform() * (xhi-xlo);
          coord[1] = ylo + random->uniform() * (yhi-ylo);
          coord[2] = zlo + random->uniform() * (zhi-zlo);
        } while (iregion->match(coord[0],coord[1],coord[2]) == 0);
      } else if (distflag == DIST_GAUSSIAN) {
        do {
          coord[0] = xmid + random->gaussian() * sigma;
          coord[1] = ymid + random->gaussian() * sigma;
          coord[2] = zmid + random->gaussian() * sigma;
        } while (iregion->match(coord[0],coord[1],coord[2]) == 0);
      } else { error->all(FLERR,"Unknown particle distribution in fix deposit");}

      if ((varflag != 0) && vartest(coord[0],coord[1],coord[2]) == 0) { continue;}

      // adjust vertical coord by offset

      if (dimension == 2) { coord[1] += offset;
      } else { coord[2] += offset;}

      // if global, reset vertical coord to be lo-hi above highest atom
      // if local, reset vertical coord to be lo-hi above highest "nearby" atom
      // local computation computes lateral distance between 2 particles w/ PBC
      // when done, have final coord of atom or center pt of molecule

      if ((globalflag != 0) || (localflag != 0)) {
        int dim = 0;
        double max{};
        double maxall{};
        double delx{};
        double dely{};
        double delz{};
        double rsq{};

        if (dimension == 2) {
          dim = 1;
          max = domain->boxlo[1];
        } else {
          dim = 2;
          max = domain->boxlo[2];
        }

        double **x = atom->x;
        int const nlocal = atom->nlocal;
        for (i = 0; i < nlocal; i++) {
          if (localflag != 0) {
            delx = coord[0] - x[i][0];
            dely = coord[1] - x[i][1];
            delz = 0.0;
            domain->minimum_image(delx,dely,delz);
            if (dimension == 2) { rsq = delx*delx;
            } else { rsq = delx*delx + dely*dely;}
            if (rsq > deltasq) { continue;}
          }
          if (x[i][dim] > max) { max = x[i][dim];}
        }

        MPI_Allreduce(&max,&maxall,1,MPI_DOUBLE,MPI_MAX,world);
        if (dimension == 2) {
          coord[1] = maxall + lo + random->uniform()*(hi-lo);
        } else {
          coord[2] = maxall + lo + random->uniform()*(hi-lo);}
      }

      // coords = coords of all atoms
      // for molecule, perform random rotation around center pt
      // apply PBC so final coords are inside box
      // also modify image flags due to PBC

      if (mode == ATOM) {
        natom = 1;
        coords[0][0] = coord[0];
        coords[0][1] = coord[1];
        coords[0][2] = coord[2];
        imageflags[0] = (static_cast<imageint>(IMGMAX) << IMG2BITS) |
          (static_cast<imageint>(IMGMAX) << IMGBITS) | IMGMAX;
      } else {
        double const rng = random->uniform();
        imol = 0;
        while (rng > molfrac[imol]) { imol++;}
        natom = onemols[imol]->natoms;
        if (dimension == 3) {
          if (orientflag != 0) {
            r[0] = rx;
            r[1] = ry;
            r[2] = rz;
          } else {
            r[0] = random->uniform() - 0.5;
            r[1] = random->uniform() - 0.5;
            r[2] = random->uniform() - 0.5;
          }
        } else {
          r[0] = r[1] = 0.0;
          r[2] = 1.0;
        }
        double const theta = random->uniform() * MY_2PI;
        MathExtra::norm3(r);
        MathExtra::axisangle_to_quat(r,theta,quat);
        MathExtra::quat_to_mat(quat,rotmat);
        for (i = 0; i < natom; i++) {
          MathExtra::matvec(rotmat,onemols[imol]->dx[i],coords[i]);
          coords[i][0] += coord[0];
          coords[i][1] += coord[1];
          coords[i][2] += coord[2];

          imageflags[i] = (static_cast<imageint>(IMGMAX) << IMG2BITS) |
            (static_cast<imageint>(IMGMAX) << IMGBITS) | IMGMAX;
          domain->remap(coords[i],imageflags[i]);
        }
      }

      // check distance between any existing atom and any inserted atom
      // if less than near, try again
      // use minimum_image() to account for PBC

      double **x = atom->x;
      int const nlocal = atom->nlocal;

      flag = 0;
      for (m = 0; m < natom; m++) {
        for (i = 0; i < nlocal; i++) {
          delx = coords[m][0] - x[i][0];
          dely = coords[m][1] - x[i][1];
          delz = coords[m][2] - x[i][2];
          domain->minimum_image(delx,dely,delz);
          rsq = delx*delx + dely*dely + delz*delz;
          if (rsq < nearsq) { flag = 1;}
        }
      }
      MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_MAX,world);
      if (flagall != 0) { continue;}

      // proceed with insertion

      nlocalprev = atom->nlocal;

      // choose random velocity for new particle
      // used for every atom in molecule

      if (tempflag != 0) {
        vnew[0] = random->gaussian() * vsigma;
        vnew[1] = random->gaussian() * vsigma;
        vnew[2] = random->gaussian() * vsigma;
      } else {
        vnew[0] = vxlo + random->uniform() * (vxhi-vxlo);
        vnew[1] = vylo + random->uniform() * (vyhi-vylo);
        vnew[2] = vzlo + random->uniform() * (vzhi-vzlo);
      }

      // if target specified, change velocity vector accordingly

      if (targetflag != 0) {
        double const vel = sqrt(vnew[0]*vnew[0] + vnew[1]*vnew[1] + vnew[2]*vnew[2]);
        delx = tx - coord[0];
        dely = ty - coord[1];
        delz = tz - coord[2];
        double const rsq = delx*delx + dely*dely + delz*delz;
        if (rsq > 0.0) {
          double const rinv = sqrt(1.0/rsq);
          vnew[0] = delx*rinv*vel;
          vnew[1] = dely*rinv*vel;
          vnew[2] = delz*rinv*vel;
        }
      }

      // check if new atoms are in my sub-box or above it if I am highest proc
      // if so, add atom to my list via create_atom()
      // initialize additional info about the atoms
      // set group mask to "all" plus fix group

      for (m = 0; m < natom; m++) {
        if (domain->triclinic != 0) {
          domain->x2lamda(coords[m],lamda);
          newcoord = lamda;
        } else { newcoord = coords[m];}

        flag = 0;
        if (newcoord[0] >= sublo[0] && newcoord[0] < subhi[0] &&
            newcoord[1] >= sublo[1] && newcoord[1] < subhi[1] &&
            newcoord[2] >= sublo[2] && newcoord[2] < subhi[2]) { flag = 1;
        } else if (dimension == 3 && newcoord[2] >= domain->boxhi[2]) {
          if (comm->layout != Comm::LAYOUT_TILED) {
            if (comm->myloc[2] == comm->procgrid[2]-1 &&
                newcoord[0] >= sublo[0] && newcoord[0] < subhi[0] &&
                newcoord[1] >= sublo[1] && newcoord[1] < subhi[1]) { flag = 1;}
          } else {
            if (comm->mysplit[2][1] == 1.0 &&
                newcoord[0] >= sublo[0] && newcoord[0] < subhi[0] &&
                newcoord[1] >= sublo[1] && newcoord[1] < subhi[1]) { flag = 1;}
          }
        } else if (dimension == 2 && newcoord[1] >= domain->boxhi[1]) {
          if (comm->layout != Comm::LAYOUT_TILED) {
            if (comm->myloc[1] == comm->procgrid[1]-1 &&
                newcoord[0] >= sublo[0] && newcoord[0] < subhi[0]) { flag = 1;}
          } else {
            if (comm->mysplit[1][1] == 1.0 &&
                newcoord[0] >= sublo[0] && newcoord[0] < subhi[0]) { flag = 1;}
          }
        }

        if (flag != 0) {
          if (mode == ATOM) {
            atom->avec->create_atom(ntype,coords[m]);
          } else {
            atom->avec->create_atom(ntype+onemols[imol]->type[m],coords[m]);
          }
          n = atom->nlocal - 1;
          atom->tag[n] = maxtag_all + m+1;
          if (mode == MOLECULE) {
            if (atom->molecule_flag != 0) {
              if (onemols[imol]->moleculeflag != 0) {
                atom->molecule[n] = maxmol_all + onemols[imol]->molecule[m];
              } else {
                atom->molecule[n] = maxmol_all+1;
              }
            }
            if (atom->molecular == Atom::TEMPLATE) {
              atom->molindex[n] = 0;
              atom->molatom[n] = m;
            }
          }
          atom->mask[n] = 1 | groupbit;
          atom->image[n] = imageflags[m];
          atom->v[n][0] = vnew[0];
          atom->v[n][1] = vnew[1];
          atom->v[n][2] = vnew[2];
          if (mode == MOLECULE) {
            onemols[imol]->quat_external = quat;
            atom->add_molecule_atom(onemols[imol],m,n,maxtag_all);
          }
          modify->create_attribute(n);

          if (groupid != 0) { atom->mask[n] |= (1 << groupid); }
          if (markflag != 0) { fix_keddf->mark(n); }
        }
      }

      // FixRigidSmall::set_molecule stores rigid body attributes
      //   coord is new position of geometric center of mol, not COM
      // FixShake::set_molecule stores shake info for molecule

      if (mode == MOLECULE) {
        if (rigidflag != 0) {
          fixrigid->set_molecule(nlocalprev,maxtag_all,imol,coord,vnew,quat);
        } else if (shakeflag != 0) {
          fixshake->set_molecule(nlocalprev,maxtag_all,imol,coord,vnew,quat);}
      }

      success = 1;
      break;
    }

    // warn if not successful b/c too many attempts

    if ((warnflag != 0) && (success == 0) && comm->me == 0) {
      error->warning(FLERR,"One or more particle depositions were unsuccessful");
      warnflag = 0;
    }

    // reset global natoms,nbonds,etc
    // increment maxtag_all and maxmol_all if necessary
    // if global map exists, reset it now instead of waiting for comm
    //   since other pre-exchange fixes may use it
    //   invoke map_init() b/c atom count has grown

    if (success != 0) {
      atom->natoms += natom;
      if (atom->natoms < 0) {
        error->all(FLERR,"Too many total atoms");}
      if (mode == MOLECULE) {
        atom->nbonds += onemols[imol]->nbonds;
        atom->nangles += onemols[imol]->nangles;
        atom->ndihedrals += onemols[imol]->ndihedrals;
        atom->nimpropers += onemols[imol]->nimpropers;
      }
      maxtag_all += natom;
      if (maxtag_all >= MAXTAGINT) {
        error->all(FLERR,"New atom IDs exceed maximum allowed ID");}
      if (mode == MOLECULE && (atom->molecule_flag != 0)) {
        if (onemols[imol]->moleculeflag != 0) {
          maxmol_all += onemols[imol]->nmolecules;
        } else {
          maxmol_all++;
        }
      }
      ninserted++;
    }

    // rebuild atom map

    if (atom->map_style != Atom::MAP_NONE) {
      atom->map_init();
      atom->map_set();
    }
  }

  // next timestep to insert
  // next_reneighbor = 0 if done

  if (ninserted < ninsert) { next_reneighbor += nfreq;
  } else { next_reneighbor = 0;}
}

/* ----------------------------------------------------------------------
   maxtag_all = current max atom ID for all atoms
   maxmol_all = current max molecule ID for all atoms
------------------------------------------------------------------------- */

void FixRegen::find_maxid()
{
  tagint *tag = atom->tag;
  tagint *molecule = atom->molecule;
  int const nlocal = atom->nlocal;

  tagint max = 0;
  for (int i = 0; i < nlocal; i++) { max = MAX(max,tag[i]);}
  MPI_Allreduce(&max,&maxtag_all,1,MPI_LMP_TAGINT,MPI_MAX,world);

  if (mode == MOLECULE && (molecule != nullptr)) {
    max = 0;
    for (int i = 0; i < nlocal; i++) { max = MAX(max,molecule[i]);}
    MPI_Allreduce(&max,&maxmol_all,1,MPI_LMP_TAGINT,MPI_MAX,world);
  }
}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line
------------------------------------------------------------------------- */

void FixRegen::options(int narg, char **arg)
{
  // defaults

  iregion = nullptr;
  idregion = nullptr;
  varflag = 0;
  vstr = xstr = ystr = zstr = nullptr;
  mode = ATOM;
  molfrac = nullptr;
  rigidflag = 0;
  idrigid = nullptr;
  shakeflag = 0;
  idshake = nullptr;
  idnext = 0;
  globalflag = localflag = 0;
  lo = hi = deltasq = 0.0;
  nearsq = 0.0;
  maxattempt = 10;
  rateflag = 0;
  vxlo = vxhi = vylo = vyhi = vzlo = vzhi = 0.0;
  distflag = DIST_UNIFORM;
  sigma = 1.0;
  xmid = ymid = zmid = 0.0;
  scaleflag = 1;
  targetflag = 0;
  orientflag = 0;
  warnflag = 1;
  rx = 0.0;
  ry = 0.0;
  rz = 0.0;
  temperature = vsigma = 0;
  tempflag = 0;
  groupid = 0;
  markflag = 0;
  fix_keddf = nullptr;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      iregion = domain->get_region_by_id(arg[iarg+1]);
      if (iregion == nullptr) { error->all(FLERR,"Region ID {} for fix deposit does not exist",arg[iarg+1]);}
      idregion = utils::strdup(arg[iarg+1]);
      iarg += 2;

    } else if (strcmp(arg[iarg], "var") == 0) {
      if (iarg + 2 > narg) { utils::missing_cmd_args(FLERR, "fix deposit var", error);}
      delete[] vstr;
      vstr = utils::strdup(arg[iarg + 1]);
      varflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg], "set") == 0) {
      if (iarg + 3 > narg) { utils::missing_cmd_args(FLERR, "fix deposit set", error);}
      if (strcmp(arg[iarg + 1], "x") == 0) {
        delete[] xstr;
        xstr = utils::strdup(arg[iarg + 2]);
      } else if (strcmp(arg[iarg + 1], "y") == 0) {
        delete[] ystr;
        ystr = utils::strdup(arg[iarg + 2]);
      } else if (strcmp(arg[iarg + 1], "z") == 0) {
        delete[] zstr;
        zstr = utils::strdup(arg[iarg + 2]);
      } else {
        error->all(FLERR, "Unknown fix deposit set option {}", arg[iarg + 2]);}
      iarg += 3;

    } else if (strcmp(arg[iarg],"mol") == 0) {
      if (iarg+2 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      int const imol = atom->find_molecule(arg[iarg+1]);
      if (imol == -1) { error->all(FLERR,"Molecule template ID for fix deposit does not exist");}
      mode = MOLECULE;
      onemols = &atom->molecules[imol];
      nmol = onemols[0]->nset;
      delete [] molfrac;
      molfrac = new double[nmol];
      molfrac[0] = 1.0/nmol;
      for (int i = 1; i < nmol-1; i++) { molfrac[i] = molfrac[i-1] + 1.0/nmol;}
      molfrac[nmol-1] = 1.0;
      iarg += 2;
    } else if (strcmp(arg[iarg],"molfrac") == 0) {
      if (mode != MOLECULE) { error->all(FLERR,"Illegal fix deposit command");}
      if (iarg+nmol+1 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      molfrac[0] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      for (int i = 1; i < nmol; i++) {
        molfrac[i] = molfrac[i-1] + utils::numeric(FLERR,arg[iarg+i+1],false,lmp);}
      if (molfrac[nmol-1] < 1.0-EPSILON || molfrac[nmol-1] > 1.0+EPSILON) {
        error->all(FLERR,"Illegal fix deposit command");}
      molfrac[nmol-1] = 1.0;
      iarg += nmol+1;
    } else if (strcmp(arg[iarg],"rigid") == 0) {
      if (iarg+2 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      delete [] idrigid;
      idrigid = utils::strdup(arg[iarg+1]);
      rigidflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"shake") == 0) {
      if (iarg+2 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      delete [] idshake;
      idshake = utils::strdup(arg[iarg+1]);
      shakeflag = 1;
      iarg += 2;

    } else if (strcmp(arg[iarg],"id") == 0) {
      if (iarg+2 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      if (strcmp(arg[iarg+1],"max") == 0) { idnext = 0;
      } else if (strcmp(arg[iarg+1],"next") == 0) { idnext = 1;
      } else { error->all(FLERR,"Illegal fix deposit command");}
      iarg += 2;
    } else if (strcmp(arg[iarg],"global") == 0) {
      if (iarg+3 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      globalflag = 1;
      localflag = 0;
      lo = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      hi = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      iarg += 3;
    } else if (strcmp(arg[iarg],"local") == 0) {
      if (iarg+4 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      localflag = 1;
      globalflag = 0;
      lo = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      hi = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      deltasq = utils::numeric(FLERR,arg[iarg+3],false,lmp) *
        utils::numeric(FLERR,arg[iarg+3],false,lmp);
      iarg += 4;

    } else if (strcmp(arg[iarg],"near") == 0) {
      if (iarg+2 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      nearsq = utils::numeric(FLERR,arg[iarg+1],false,lmp) *
        utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"temp") == 0) {
      if (iarg+2 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      temperature = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      tempflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"group") == 0) {
      if (iarg+2 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      groupid = group->find(arg[iarg+1]);
      if (groupid <= 0) { error->all(FLERR,"Specified group not found or group all is used");}
      iarg += 2;
    } else if (strcmp(arg[iarg],"mark") == 0) {
      if (iarg+2 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      fix_keddf = dynamic_cast<FixKedff *>(modify->get_fix_by_id(arg[iarg+1]));
      if (fix_keddf == nullptr) { error->all(FLERR,"Specified FixCapture does not exist: {}", arg[iarg+1]);}
      markflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"attempt") == 0) {
      if (iarg+2 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      maxattempt = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"rate") == 0) {
      if (iarg+2 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      rateflag = 1;
      rate = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"vx") == 0) {
      if (iarg+3 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      vxlo = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      vxhi = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      iarg += 3;
    } else if (strcmp(arg[iarg],"vy") == 0) {
      if (iarg+3 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      vylo = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      vyhi = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      iarg += 3;
    } else if (strcmp(arg[iarg],"vz") == 0) {
      if (iarg+3 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      vzlo = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      vzhi = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      iarg += 3;
    } else if (strcmp(arg[iarg],"orient") == 0) {
      if (iarg+4 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      orientflag = 1;
      rx = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      ry = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      rz = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      if (domain->dimension == 2 && (rx != 0.0 || ry != 0.0)) {
        error->all(FLERR,"Illegal fix deposit orient settings");}
      if (rx == 0.0 && ry == 0.0 && rz == 0.0) {
        error->all(FLERR,"Illegal fix deposit orient settings");}
      iarg += 4;
    } else if (strcmp(arg[iarg],"units") == 0) {
      if (iarg+2 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      if (strcmp(arg[iarg+1],"box") == 0) { scaleflag = 0;
      } else if (strcmp(arg[iarg+1],"lattice") == 0) { scaleflag = 1;
      } else { error->all(FLERR,"Illegal fix deposit command");}
      iarg += 2;
    } else if (strcmp(arg[iarg],"gaussian") == 0) {
      if (iarg+5 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      xmid = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      ymid = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      zmid = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      sigma = utils::numeric(FLERR,arg[iarg+4],false,lmp);
      distflag = DIST_GAUSSIAN;
      iarg += 5;
    } else if (strcmp(arg[iarg],"target") == 0) {
      if (iarg+4 > narg) { error->all(FLERR,"Illegal fix deposit command");}
      tx = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      ty = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      tz = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      targetflag = 1;
      iarg += 4;
    } else { error->all(FLERR,"Illegal fix deposit command");}
  }

  // error check and further setup for variable test

  if ((vstr == nullptr) && ((xstr != nullptr) || (ystr != nullptr) || (zstr != nullptr))) {
    error->all(FLERR, "Incomplete use of variables in fix deposit command");}
  if ((vstr != nullptr) && ((xstr == nullptr) && (ystr == nullptr) && (zstr == nullptr))) {
    error->all(FLERR, "Incomplete use of variables in fix deposit command");}

  if (varflag != 0) {
    vvar = input->variable->find(vstr);
    if (vvar < 0) { error->all(FLERR, "Variable {} for fix deposit does not exist", vstr);}
    if (input->variable->equalstyle(vvar) == 0) {
      error->all(FLERR, "Variable for fix deposit is invalid style");}

    if (xstr != nullptr) {
      xvar = input->variable->find(xstr);
      if (xvar < 0) { error->all(FLERR, "Variable {} for fix deposit does not exist", xstr);}
      if (input->variable->internalstyle(xvar) == 0) {
        error->all(FLERR, "Variable for fix deposit is invalid style");}
    }
    if (ystr != nullptr) {
      yvar = input->variable->find(ystr);
      if (yvar < 0) { error->all(FLERR, "Variable {} for fix deposit does not exist", ystr);}
      if (input->variable->internalstyle(yvar) == 0) {
        error->all(FLERR, "Variable for fix deposit is invalid style");}
    }
    if (zstr != nullptr) {
      zvar = input->variable->find(zstr);
      if (zvar < 0) { error->all(FLERR, "Variable {} for fix deposit does not exist", zstr);}
      if (input->variable->internalstyle(zvar) == 0) {
        error->all(FLERR, "Variable for fix deposit is invalid style");}
    }
  }
}

/* ----------------------------------------------------------------------
   output number of successful insertions
------------------------------------------------------------------------- */

double FixRegen::compute_scalar()
{
  return ninserted;}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixRegen::write_restart(FILE *fp)
{
  int n = 0;
  double list[5];
  list[n++] = random->state();
  list[n++] = ninserted;
  list[n++] = ubuf(nfirst).d;
  list[n++] = ubuf(next_reneighbor).d;
  list[n++] = ubuf(update->ntimestep).d;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixRegen::restart(char *buf)
{
  int n = 0;
  auto *list = reinterpret_cast<double *>(buf);

  seed = static_cast<int>(list[n++]);
  ninserted = static_cast<int>(list[n++]);
  nfirst = static_cast<bigint>(ubuf(list[n++]).i);
  next_reneighbor = static_cast<bigint>(ubuf(list[n++]).i);

  bigint const ntimestep_restart = static_cast<bigint>(ubuf(list[n++]).i);
  if (ntimestep_restart != update->ntimestep) {
    error->all(FLERR,"Must not reset timestep when restarting this fix");}

  random->reset(seed);}

/* ----------------------------------------------------------------------
   extract particle radius for atom type = itype
------------------------------------------------------------------------- */

void *FixRegen::extract(const char *str, int &itype)
{
  if (strcmp(str,"radius") == 0) {
    if (mode == ATOM) {
      if (itype == ntype) { oneradius = 0.5;
      } else { oneradius = 0.0;}

    } else {

      // loop over onemols molecules
      // skip a molecule with no atoms as large as itype

      oneradius = 0.0;
      for (int i = 0; i < nmol; i++) {
        if (itype > ntype+onemols[i]->ntypes) { continue;}
        double *radius = onemols[i]->radius;
        int *type = onemols[i]->type;
        int const natoms = onemols[i]->natoms;

        // check radii of atoms in Molecule with matching types
        // default to 0.5, if radii not defined in Molecule
        //   same as atom->avec->create_atom(), invoked in pre_exchange()

        for (int i = 0; i < natoms; i++) {
          if (type[i]+ntype == itype) {
            if (radius != nullptr) { oneradius = MAX(oneradius,radius[i]);
            } else { oneradius = MAX(oneradius,0.5);}
          }
}
      }
    }
    itype = 0;
    return &oneradius;
  }

  return nullptr;}

/* ----------------------------------------------------------------------
   test a generated atom position against variable evaluation
   first set x,y,z values in internal variables
------------------------------------------------------------------------- */

int FixRegen::vartest(double x, double y, double z)
{
  if (xstr != nullptr) { input->variable->internal_set(xvar, x);}
  if (ystr != nullptr) { input->variable->internal_set(yvar, y);}
  if (zstr != nullptr) { input->variable->internal_set(zvar, z);}

  double const value = input->variable->compute_equal(vvar);

  if (value == 0.0) { return 0;}
  return 1;}
