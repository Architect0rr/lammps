#ifndef __NUCC_CUSTOM_STL_ALLOCATOR_HPP
#define __NUCC_CUSTOM_STL_ALLOCATOR_HPP

#include "memory.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

namespace NUCC {

class MemoryKeeper {
 public:
  MemoryKeeper() = delete;
  MemoryKeeper(const MemoryKeeper &) = delete;
  MemoryKeeper(MemoryKeeper &&) = delete;
  MemoryKeeper &operator=(const MemoryKeeper &) = delete;
  MemoryKeeper &operator=(MemoryKeeper &&) = delete;

  MemoryKeeper(Memory *mem) : mem(mem) {}
  ~MemoryKeeper() { clear(); }

  template <typename T>
  void store(T *&ptr, const size_t size)
  {
    infos.emplace_back(ptr, size);
  }

  void clear()
  {
    for (auto &pool : infos) { mem->destroy(pool.ptr); }
  }

  uint64_t pool_size() const noexcept
  {
    assert(_pool_size > 0);
    return _pool_size;
  }

  void pool_size(uint64_t n) noexcept { _pool_size = n; }

 private:
  struct PoolInfo {
    template <typename T>
    PoolInfo(T *&ptr, const size_t size) : ptr(reinterpret_cast<void *>(ptr)), size(size * sizeof(T))
    {
    }

    PoolInfo() : ptr(nullptr), size(0) {};

    void *ptr = nullptr;
    size_t size = 0;
  };

  Memory *mem = nullptr;
  uint64_t _pool_size = 0;
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

  // Constructor
  CustomAllocator(Memory *const memory, MemoryKeeper *const keeper) : memory_(memory), keeper_(keeper) {}

  // Copy constructor
  template <typename U>
  CustomAllocator(const CustomAllocator<U> &other) : memory_(other.memory_), keeper_(other.keeper_) {}

  ~CustomAllocator() {}

  // Allocate memory
  T *allocate(const std::size_t n)
  {
    T *ptr = nullptr;
    uint64_t pool_size = keeper_->pool_size();
    if (n > pool_size) {
      // If requested size is larger than pool, allocate separately
      memory_->create<T>(ptr, n, "CustomAllocator_Large");
      keeper_->store(ptr, n);
      return ptr;
    }
    if (current == nullptr || left < n) {
      // Pool is full or not initialized, request a new pool
      memory_->create<T>(current, pool_size, "CustomAllocator_Pool");
      left = pool_size;
    }
    ptr = current;
    left -= n;
    current += n;
    return ptr;
  }

  // Deallocate memory
  void deallocate(T *p, const std::size_t n) const noexcept
  {
    // Deallocation can be handled when the allocator is destroyed
    // For pool allocator, individual deallocations are often no-ops
  }

  // Equality operators
  template <typename U>
  bool operator==(const CustomAllocator<U> &other) const noexcept
  {
    return (keeper_ == other.keeper_) && (memory_ == other.memory_) && (current == other.current) && (left == other.left);
  }

  template <typename U>
  bool operator!=(const CustomAllocator<U> &other) const noexcept
  {
    return !(*this == other);
  }

 private:
  T *current = nullptr;
  size_t left = 0;

  MemoryKeeper *const keeper_;
  Memory *const memory_;
};

}    // namespace NUCC

#endif    // CUSTOM_ALLOCATOR_HPP
