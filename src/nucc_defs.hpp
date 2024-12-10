#ifndef __NUCC_DEFS_HPP
#define __NUCC_DEFS_HPP

#include "nucc_allocator.hpp"
#include "nucc_cspan.hpp"

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

}    //  namespace NUCC

#endif    // !__NUCC_DEFS_HPP
