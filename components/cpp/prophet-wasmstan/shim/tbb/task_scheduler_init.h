#ifndef __TBB_task_scheduler_init_H
#define __TBB_task_scheduler_init_H

#include <tbb/task_scheduler_observer.h>

namespace tbb {

typedef size_t stack_size_type;

class task_scheduler_init {
  public:
  task_scheduler_init(size_t, size_t) {}
};

}

#endif
