#ifdef COMMAND_CLASS
// clang-format off
CommandStyle(print_box,PrintBox);
// clang-format on
#else

#ifndef LMP_PRINT_BOX_H
#define LMP_PRINT_BOX_H

#include "command.h"

namespace LAMMPS_NS {

class PrintBox : public Command {
 public:
  PrintBox(class LAMMPS *);
  void command(int, char **) override;
 private:

};


} // namespace LAMMPS_NS

#endif
#endif
