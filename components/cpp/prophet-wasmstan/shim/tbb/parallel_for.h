#ifndef __TBB_parallel_for_H
#define __TBB_parallel_for_H

#include "partitioner.h"
#include "blocked_range.h"

namespace tbb {

template<typename RangeType, typename Body>
void parallel_for(const blocked_range<RangeType>& range, const Body& body, const simple_partitioner& partitioner) {
    for (RangeType idx = range.begin(); idx < range.end(); idx++) {
        body(idx);
    }
}

}

#endif
