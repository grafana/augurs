#ifndef __TBB_task_scheduler_observer_H
#define __TBB_task_scheduler_observer_H

namespace tbb {

class task_scheduler_observer {
  public:
  void observe(bool) {}
};

}

#endif
