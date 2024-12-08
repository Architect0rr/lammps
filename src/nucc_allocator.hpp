#ifndef __NUCC_CUSTOM_STL_ALLOCATOR_HPP
#define __NUCC_CUSTOM_STL_ALLOCATOR_HPP

#include "memory.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

namespace NUCC {

class MemoryKeeper {
 private:
  struct PoolInfo {
    template <typename T>
    constexpr PoolInfo(T*& ptr, const size_t size) noexcept : ptr(reinterpret_cast<void*>(ptr)), size(size * sizeof(T))
    {
    }

    PoolInfo() = delete;

    void* ptr = nullptr;
    size_t size = 0;
  };
 public:
  MemoryKeeper() = delete;
  MemoryKeeper(const MemoryKeeper&) = delete;
  MemoryKeeper(MemoryKeeper&&) = delete;
  MemoryKeeper& operator=(const MemoryKeeper&) = delete;
  MemoryKeeper& operator=(MemoryKeeper&&) = delete;

  MemoryKeeper(LAMMPS_NS::Memory* memory) noexcept : memory_(memory) {}
  ~MemoryKeeper() noexcept(noexcept(clear())) { clear(); }

  template <typename T>
  void store(T*& ptr, const size_t size) noexcept(noexcept(std::declval<std::vector<PoolInfo>>().emplace_back(ptr, size)))
  {
    infos.emplace_back(ptr, size);
  }

  void clear() noexcept(noexcept(std::declval<LAMMPS_NS::Memory>().destroy<void>(std::declval<void*&>())))
  {
    for (auto& pool : infos) { memory_->destroy(pool.ptr); }
  }

  inline constexpr std::size_t pool_size() const noexcept
  {
    assert(_pool_size > 0);
    return _pool_size;
  }

  template <typename T>
  inline constexpr void pool_size(std::size_t n) noexcept { _pool_size = n * sizeof(T); }

  inline constexpr void pool_size(std::size_t n) noexcept { _pool_size = n; }

  template <typename T>
  T* allocate(const std::size_t n)
  {
    T* ptr = nullptr;
    std::size_t pool_size_T = _pool_size / sizeof(T) + 1;
    if (n > pool_size_T) {
      // If requested size is larger than pool, allocate separately
      memory_->create<T>(ptr, n, "CustomAllocator_Large");
      infos.emplace_back(ptr, n);
      return ptr;
    }
    std::size_t nbytes = n * sizeof(T);
    T* _current = reinterpret_cast<T*>(current);
    if ((current == nullptr) || (left < nbytes)) {
      // Pool is full or not initialized, request a new pool
      memory_->create(_current, pool_size_T, "CustomAllocator_Pool");
      infos.emplace_back(_current, pool_size_T);
      left = _pool_size;
    }
    ptr = _current;
    left -= nbytes;
    _current += n;
    current = reinterpret_cast<void *>(_current);
    return ptr;
  }

 private:
  void* current = nullptr;
  std::size_t left = 0;
  LAMMPS_NS::Memory* const memory_ = nullptr;
  std::size_t _pool_size = 0;
  std::vector<PoolInfo> infos;
};

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

template <typename T>
class CustomAllocator {
 public:
  using value_type = T;

  template <typename U>
  friend class CustomAllocator;

  CustomAllocator() = delete;

  constexpr CustomAllocator(MemoryKeeper* const keeper) noexcept : keeper_(keeper) {}

  template <typename U>
  constexpr CustomAllocator(const CustomAllocator<U>& other) noexcept : keeper_(other.keeper_)
  {
  }

  inline T* allocate(std::size_t n) const { return keeper_->allocate<T>(n); }

  inline constexpr void deallocate(T* /*p*/, const std::size_t /*n*/) const noexcept
  {
    // Deallocation can be handled when the allocator is destroyed
    // For pool allocator, individual deallocations are often no-ops
  }

  // Equality operators
  template <typename U>
  inline constexpr bool operator==(const CustomAllocator<U>& other) const noexcept
  {
    return keeper_ == other.keeper_;
  }

  template <typename U>
  inline constexpr bool operator!=(const CustomAllocator<U>& other) const noexcept
  {
    return !(*this == other);
  }

 private:
  MemoryKeeper* const keeper_;
};

}    // namespace NUCC

#endif    // CUSTOM_ALLOCATOR_HPP
