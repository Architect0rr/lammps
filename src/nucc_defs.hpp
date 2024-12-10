#ifndef __NUCC_DEFS_HPP
#define __NUCC_DEFS_HPP

#include "nucc_allocator.hpp"

#include <scoped_allocator>
#include <unordered_map>
#include <vector>

#    define LMP_NUCC_ALLOC_COEFF 1.2
#    define LMP_NUCC_CLUSTER_MAX_OWNERS 128
#    define LMP_NUCC_CLUSTER_MAX_SIZE 300
#    define LMP_NUCC_CLUSTER_MAX_GHOST 300

#define __NUCC_CSPAN_CHECK_ACCESS
#define __NUCC_CHECK_ACCESS
// #define __NUCC_CSPAN_DEBUG_CALLS

namespace NUCC {

template <typename A>
using VecAlloc_t = CustomAllocator<A>;

template <typename A>
using Vec_t = std::vector<A, std::scoped_allocator_adaptor<VecAlloc_t<A>>>;

template <typename A, typename B>
using MapMember_t = std::pair<const A, B>;

template <typename A, typename B>
using MapAlloc_t = CustomAllocator<MapMember_t<A, B>>;

template <typename A, typename B>
using Map_t = std::unordered_map<A, B, std::hash<A>, std::equal_to<A>, std::scoped_allocator_adaptor<MapAlloc_t<A, B>>>;

struct cluster_data {
  explicit cluster_data(const int _clid) : clid(_clid) {}

  void rearrange() noexcept { ::memcpy(_atoms + l_size, _ghost, nghost * sizeof(int)); }

  NUCC::cspan<const int> atoms_all() const { return std::span<const int>(_atoms, l_size + nghost); }

  template <bool Protect = true>
    requires(!Protect)
  NUCC::cspan<int, LMP_NUCC_CLUSTER_MAX_SIZE> atoms()
  {
    return std::span<int, LMP_NUCC_CLUSTER_MAX_SIZE>(_atoms, LMP_NUCC_CLUSTER_MAX_SIZE);
  }

  NUCC::cspan<const int> atoms() const { return std::span<const int>(_atoms, l_size); }

  NUCC::cspan<int, LMP_NUCC_CLUSTER_MAX_GHOST> ghost_initial() { return std::span<int, LMP_NUCC_CLUSTER_MAX_GHOST>(_ghost, nghost); }

  NUCC::cspan<const int> ghost() const { return std::span<const int>(_atoms + l_size, nghost); }

  template <bool Protect = true>
    requires(!Protect)
  NUCC::cspan<int, LMP_NUCC_CLUSTER_MAX_OWNERS> owners()
  {
    return std::span<int, LMP_NUCC_CLUSTER_MAX_OWNERS>(_owners, LMP_NUCC_CLUSTER_MAX_OWNERS);
  }

  NUCC::cspan<const int> owners() const { return std::span<const int>(_owners, nowners); }

  int clid = 0;       // cluster ID
  int l_size = 0;     // local size
  int g_size = 0;     // global size
  int host = -1;      // host proc (me if <0)
  int nhost = 0;      // local cluster size of host proc
  int nowners = 0;    // number of owners
  int nghost = 0;     // number of ghost atoms in cluster
 private:
  int _owners[LMP_NUCC_CLUSTER_MAX_OWNERS];    // procs owning some cluster's atoms
  int _atoms[LMP_NUCC_CLUSTER_MAX_SIZE];       // local ids of atoms
  int _ghost[LMP_NUCC_CLUSTER_MAX_GHOST];      // local ids of ghost atoms
};

}    //  namespace NUCC

#endif    // !__NUCC_DEFS_HPP
