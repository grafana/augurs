// Stub out calls to exception-related functions until WASI supports them.
#include <iostream>
#include <stdlib.h>
#include <typeinfo>

extern "C" {
void *__cxa_allocate_exception(size_t) {
  std::cerr << "Exception thrown" << std::endl;
  abort();
};

void __cxa_throw(void *thrown_exception, std::type_info *tinfo,
                 void(_LIBCXXABI_DTOR_FUNC)(void *)) {
  std::cerr << "Exception thrown: " << tinfo->name() << std::endl;
  abort();
}
}
