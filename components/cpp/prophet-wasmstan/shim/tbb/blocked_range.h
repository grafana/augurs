#ifndef __TBB_blocked_range_H
#define __TBB_blocked_range_H

namespace tbb {

template<typename Value>
class blocked_range {
  public:
  blocked_range(Value begin, Value end, std::size_t grainsize)
   : begin_(begin), end_(end) {}
  Value begin() const { return begin_; }
  Value end() const { return end_; }
  std::size_t size() const { return end_ - begin_; }
  bool empty() const { return !(begin_ < end_); }
  private:
  Value begin_;
  Value end_;
};

}

#endif
