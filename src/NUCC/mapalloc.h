#include "error.h"
#include "memory.h"
#include "lmptype.h"

using namespace LAMMPS_NS;

/* The following code is modified example from the book
 * "The C++ Standard Library - A Tutorial and Reference"
 * by Nicolai M. Josuttis, Addison-Wesley, 1999
 *
 * (C) Copyright Nicolai M. Josuttis 1999.
 * Permission to copy, use, modify, sell and distribute this software
 * is granted provided this copyright notice appears in all copies.
 * This software is provided "as is" without express or implied
 * warranty, and with no claim as to its suitability for any purpose.
 */

#include <limits>
#include <iostream>

namespace NucC {
    template <class T>
    class Alloc {
      public:
        // type definitions
        typedef T        value_type;
        typedef T*       pointer;
        typedef const T* const_pointer;
        typedef T&       reference;
        typedef const T& const_reference;
        typedef bigint   size_type;
        typedef std::ptrdiff_t difference_type;

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
        Alloc(Memory* memory, Error* error) noexcept(true): memory(memory), error(error), allocated(0) {}
        Alloc(const Alloc& other) noexcept(true): memory(other.memory), error(other.error), allocated(0) {}
        Alloc(Alloc&& other) noexcept(true): memory(other.memory), error(other.error), allocated(other.allocated) {
            other.allocated = 0;
        }
        Alloc& operator=(const Alloc& other) noexcept(true) {
            memory = other.memory;
            error = other.error;
            allocated = 0;
        }
        Alloc& operator=(Alloc&& other) noexcept(true) {
            memory = other.memory;
            error = other.error;
            allocated = other.allocated;
            other.allocated = 0;
        }

        template <class U>
        Alloc (const Alloc<U>& other) noexcept(true) {
            memory = other.memory;
            error = other.error;
            allocated = 0;
        }

        template <class U>
        Alloc (Alloc<U>&& other) noexcept(true) = delete;

        ~Alloc() noexcept(true) {
            if (allocated > 0){
                error->warning(FLERR, "Destructing allocator with allocated memory: {} elements unfreed ({} bytes)", allocated, allocated*sizeof(T));
            }
        }

        // return maximum number of elements that can be allocated
        size_type max_size () const throw() {
            return std::numeric_limits<size_type>::max() / sizeof(T);
        }

        // allocate but don't initialize num elements of type T
        pointer allocate (size_type num, const void* = 0) {
            static size_type ncall = 0;
            static char name[33];
            snprintf(name, 33, "mapalloc_%d", ncall);
            ++ncall;
            allocated += num;
            return static_cast<pointer>(memory->smalloc(num*sizeof(T), name));
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
            allocated -= num;
            memory->sfree(p);
        }

      private:
        Memory* memory;
        Error* error;
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
