// Prevents the compiler from optimizing everything away.
template <class T> void DoNotOptimize(const T &var) {
  asm volatile("" : "+m"(const_cast<T &>(var)));
}

// Writes a single double with inconsistent shadow to v.
void CreateInconsistency(double *data) {
  double num = 0.6;
  double denom = 0.2;
  // Prevent the compiler from constant-folding this.
  DoNotOptimize(num);
  DoNotOptimize(denom);
  // Both values are very close to 0.0, but shadow value is closer.
  *data = 1.0 / (num / denom - 3.0);
}
