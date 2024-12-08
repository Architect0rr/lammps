#ifndef __NUCC_DEFS_HPP
#define __NUCC_DEFS_HPP

#include "nucc_allocator.hpp"

#include <scoped_allocator>
#include <unordered_map>
#include <vector>

namespace NUCC {

template <typename A, typename B>
using MapAlloc_t = std::scoped_allocator_adaptor<CustomAllocator<std::pair<const A, B>>>;

template <typename A>
using vec_t = std::vector<A, std::scoped_allocator_adaptor<CustomAllocator<A>>>;

template <typename A, typename B>
using UMap_t = std::unordered_map<A, B, std::hash<A>, std::equal_to<A>, MapAlloc_t<A, B>>;

}    //  namespace NUCC

#endif    // !__NUCC_DEFS_HPP
