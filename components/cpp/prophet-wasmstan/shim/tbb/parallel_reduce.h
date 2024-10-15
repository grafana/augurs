#ifndef __TBB_parallel_reduce_H
#define __TBB_parallel_reduce_H

#include "partitioner.h"

namespace tbb {

template<typename Range, typename Body>
void parallel_reduce(const Range& range, Body& body) {
  body(range);
}

template<typename Range, typename Body>
void parallel_deterministic_reduce(const Range& range, Body& body,
                     const simple_partitioner& partitioner) {
  body(range);
}

}

#endif
