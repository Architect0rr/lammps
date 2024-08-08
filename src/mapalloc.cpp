#include "mapalloc.h"

#include <iomanip>

// Cantor pairing function
uint64_t NucC::cantor(const int64_t x, const int64_t y)
{
  return (x + y) * (x + y + 1) / 2 + y;
}

// Hash function for integer pairs
std::string NucC::hashPair(const int64_t a, const int64_t b)
{
  constexpr uint64_t LARGE_PRIME = 2147483647;
  // Calculate hash using Cantor pairing and prime multiplication
  std::stringstream ss;
  ss << std::hex << std::setfill('0') << std::setw(32) << cantor(a, b) * LARGE_PRIME;
  return ss.str();
}