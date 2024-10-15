#include <sstream>
#include <string>
#include <vector>

#include <stan/callbacks/writer.hpp>

// An implementation of stan::callbacks::writer that stores the
// names and values written to the stream into vectors.
class StructuredWriter : public stan::callbacks::writer {
public:
  StructuredWriter(const std::string &comment_prefix = "#")
      : comment_prefix(comment_prefix) {}

  void operator()(const std::vector<std::string> &names) {
    this->names = names;
  }

  void operator()(const std::vector<double> &values) {
    this->values.push_back(values);
  }

  void operator()() { comment << comment_prefix << std::endl; }
  void operator()(const std::string &message) {
    comment << comment_prefix << message << std::endl;
  }

  std::vector<std::string> get_names() { return names; }
  std::vector<std::vector<double>> get_values() { return values; }

  std::string get_comment() { return comment.str(); }

private:
  std::vector<std::string> names;
  std::vector<std::vector<double>> values;
  std::stringstream comment;
  std::string comment_prefix;
};
