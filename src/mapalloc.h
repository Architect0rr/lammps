#ifndef NUCC_MAPALLOC_H
#define NUCC_MAPALLOC_H

#include "error.h"
#include "memory.h"
#include "lmptype.h"
#include "comm.h"

#include <unordered_map>
#include <limits>
#include <iostream>
#include <cstdint>
#include <iomanip>
#include <sstream>

using namespace LAMMPS_NS;

namespace NucC {
    // Cantor pairing function
    uint64_t cantor(const int64_t x, const int64_t y);

    // Hash function for integer pairs
    std::string hashPair(const int64_t a, const int64_t b);

    class Mgr{
      public:
        int getID(){
            return id++;
        }

        int64_t add(int64_t n){
            return allocated += n;
        }

      private:
        int64_t allocated = 0;
        int id = 0;
    };

    template <class T>
    class Alloc {
      public:
        // type definitions
        typedef T        value_type;
        typedef T*       pointer;
        typedef const T* const_pointer;
        typedef T&       reference;
        typedef const T& const_reference;
        typedef LAMMPS_NS::bigint   size_type;
        typedef std::ptrdiff_t difference_type;
        static constexpr unsigned type_size = sizeof(T);

        template <typename U>
        friend class Alloc;

        // rebind allocator to type U
        template <class U>
        struct rebind {
            typedef Alloc<U> other;
        };

        // return address of values
        pointer address (reference value) const {
            return &value;
        }
        const_pointer address (const_reference value) const {
            return &value;
        }

        // constructors and destructor
        Alloc() = delete;
        Alloc(LAMMPS_NS::LAMMPS* lmp) noexcept(true): lmp(lmp), allocated(0), ncall(0) {}
        Alloc(const Alloc& other) noexcept(true): lmp(lmp), allocated(0), ncall(0) {}
        Alloc(Alloc&& other) noexcept(true): lmp(other.lmp), allocated(other.allocated), ncall(other.ncall) {
            other.allocated = 0;
            other.ncall = 0;
        }
        Alloc& operator=(const Alloc& other) noexcept(true) {
            lmp = other.lmp;
            allocated = 0;
            ncall = 0;
        }
        Alloc& operator=(Alloc&& other) noexcept(true) {
            lmp = other.lmp;
            allocated = other.allocated;
            other.allocated = 0;
            ncall = other.ncall;
            other.ncall = 0;
        }

        template <class U>
        Alloc (const Alloc<U>& other) noexcept(true): lmp(other.lmp), allocated(0), ncall(0) {}

        template <class U>
        Alloc (Alloc<U>&& other) noexcept(true) = delete;

        ~Alloc() noexcept(true) {
            if (allocated > 0){
                lmp->error->warning(FLERR, "Destructing allocator with allocated memory: {} elements unfreed ({} bytes)", allocated, allocated*type_size);
            }
        }

        // return maximum number of elements that can be allocated
        size_type max_size () const throw() {
            return std::numeric_limits<size_type>::max() / type_size;
        }

        // allocate but don't initialize num elements of type T
        pointer allocate (volatile size_type num, const void* = 0) {
            volatile size_type nc = num * type_size;
            allocated += nc;
            // const char *cname = fmt::format("mapalloc_{}_{}", hashPair(type_size*ncall, lmp->comm->me), ncall++).c_str();
            return static_cast<pointer>(lmp->memory->smalloc(nc, "cname"));
        }

        // initialize elements of allocated storage p with value value
        void construct (pointer p, const T& value) {
            // initialize memory with placement new
            new((void*)p)T(value);
        }

        // destroy elements of initialized storage p
        void destroy (pointer p) {
            // destroy objects by calling their destructor
            p->~T();
        }

        // deallocate storage p of deleted elements
        void deallocate (pointer p, size_type num) {
            size_type nc = num*type_size;
            allocated -= nc;
            lmp->memory->sfree(p);
        }

      private:
        size_type ncall = 0;
        LAMMPS_NS::LAMMPS* lmp = nullptr;
        size_type allocated = 0;
    };

    // return that all specializations of this allocator are interchangeable
    template <class T1, class T2>
    bool operator== (const Alloc<T1>&,
                     const Alloc<T2>&) throw() {
        return true;
    }
    template <class T1, class T2>
    bool operator!= (const Alloc<T1>&,
                     const Alloc<T2>&) throw() {
        return false;
    }
} // namespace NucC

#endif // !NUCC_MAPALLOC_H
