// Note: this function comes from https://stackoverflow.com/questions/30401486/ecma6-generators-yield-promise.
// It is used to convert a generator function into a promise.
// `jco transpile` generates a similar function but it didn't work for me.
// I'm not sure why, but I'll raise an issue on the `jco` repo.
// See the `justfile` for how this gets shimmed into the transpiled code;
// in short, we use `ripgrep` as in
// https://unix.stackexchange.com/questions/181180/replace-multiline-string-in-files
// (it was a Stack-Overflow heavy day...)
// The indentation is intentional so the function matches the original.
  function run(g) {
    return Promise.resolve(function step(v) {
      const res = g.next(v);
      if (res.done) return res.value;
      return res.value.then(step);
    }());
  }
  return run(gen);
