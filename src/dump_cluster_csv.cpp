#include "dump_cluster_csv.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

DumpClusterCSV::DumpClusterCSV(LAMMPS *lmp, int narg, char **arg) :
    Dump(lmp, narg, arg), num_vectors(0), num_scalars(0)
{
    // Constructor implementation
    if (narg < 6) {
        error->all(FLERR, "Illegal dump cluster/csv command");
    }

    // Parse arguments
    size_cutoff = utils::inumeric(FLERR, arg[5], true, lmp);
    if (size_cutoff < 1) {
        error->all(FLERR, "size_cutoff for dump cluster/csv must be greater than 0");
    }

    // Initialize file pointers
    create_ptr_array(file_vectors, num_vectors, "vector_files");
    file_scalars = fopen("scalars.csv", "a");
    if (file_scalars == nullptr) {
        error->one(FLERR, "Cannot open file scalars.csv");
    }
}

/* ---------------------------------------------------------------------- */

DumpClusterCSV::~DumpClusterCSV() noexcept(true)
{
    // Destructor implementation
    for (int i = 0; i < num_vectors; ++i) {
        if (file_vectors[i] != nullptr) {
            fclose(file_vectors[i]);
        }
    }
    if (file_scalars != nullptr) {
        fclose(file_scalars);
    }
}

/* ---------------------------------------------------------------------- */

void DumpClusterCSV::init_style()
{
    // Initialization code
    if (comm->me == 0) {
        fprintf(file_scalars, "ntimestep");
        for (int i = 0; i < num_scalars; ++i) {
            fprintf(file_scalars, ",scalar_%d", i);
        }
        fprintf(file_scalars, "\n");
    }
}

/* ---------------------------------------------------------------------- */

void DumpClusterCSV::write()
{
    // Code to write data to CSV
    if (comm->me == 0) {
        fprintf(file_scalars, "%d", update->ntimestep);
        for (int i = 0; i < num_scalars; ++i) {
            fprintf(file_scalars, ",%f", compute_scalars[i]->scalar);
        }
        fprintf(file_scalars, "\n");
        fflush(file_scalars);
    }
}
