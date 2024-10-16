#ifndef __TBB_task_arena_H
#define __TBB_task_arena_H

namespace tbb {
namespace this_task_arena {

template<typename F>
auto isolate(const F& f) {
    return f();
}

}
}

#endif
