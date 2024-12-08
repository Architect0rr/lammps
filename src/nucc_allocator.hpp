#ifndef __NUCC_CUSTOM_STL_ALLOCATOR_HPP
#define __NUCC_CUSTOM_STL_ALLOCATOR_HPP

#include "memory.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

namespace NUCC {

template <typename T>
class MemoryKeeper {
 public:
  MemoryKeeper() = delete;
  MemoryKeeper(const MemoryKeeper&) = delete;
  MemoryKeeper(MemoryKeeper&&) = delete;
  MemoryKeeper& operator=(const MemoryKeeper&) = delete;
  MemoryKeeper& operator=(MemoryKeeper&&) = delete;

  constexpr MemoryKeeper(Memory* memory) noexcept : memory_(memory) {}
  ~MemoryKeeper() noexcept(noexcept(clear())) { clear(); }

  void store(T*& ptr, const size_t size) noexcept(noexcept(infos.emplace_back(ptr, size)))
  {
    infos.emplace_back(ptr, size);
  }

  void clear() noexcept(noexcept(std::declval<Memory>().destroy<void>(std::declval<void*&>())))
  {
    for (auto& pool : infos) { memory_->destroy(pool.ptr); }
  }

  inline constexpr std::size_t pool_size() const noexcept
  {
    assert(_pool_size > 0);
    return _pool_size;
  }

  inline constexpr void pool_size(std::size_t n) noexcept { _pool_size = n; }

  T* allocate(const std::size_t n)
  {
    T* ptr = nullptr;
    if (n > _pool_size) {
      // If requested size is larger than pool, allocate separately
      memory_->create<T>(ptr, n, "CustomAllocator_Large");
      infos.emplace_back(ptr, n);
      return ptr;
    }
    if (current == nullptr || left < n) {
      // Pool is full or not initialized, request a new pool
      memory_->create<T>(current, _pool_size, "CustomAllocator_Pool");
      infos.emplace_back(current, _pool_size);
      left = _pool_size;
    }
    ptr = current;
    left -= n;
    current += n;
    return ptr;
  }

 private:
  struct PoolInfo {
    constexpr PoolInfo(T*& ptr, const size_t size) noexcept : ptr(reinterpret_cast<void*>(ptr)), size(size * sizeof(T))
    {
    }

    PoolInfo() = delete;

    void* ptr = nullptr;
    size_t size = 0;
  };

  T* current = nullptr;
  std::size_t left = 0;
  Memory* const memory_ = nullptr;
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

  constexpr CustomAllocator(MemoryKeeper<T>* const keeper) noexcept : keeper_(keeper) {}

  template <typename U>
  constexpr CustomAllocator(const CustomAllocator<U>& other) noexcept : keeper_(other.keeper_)
  {
  }

  inline T* allocate(const std::size_t n) const { return keeper_->allocate(n); }

  inline constexpr void deallocate(T /**p*/, const std::size_t /*n*/) const noexcept
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
  MemoryKeeper<T>* const keeper_;
};

}    // namespace NUCC

#endif    // CUSTOM_ALLOCATOR_HPP
