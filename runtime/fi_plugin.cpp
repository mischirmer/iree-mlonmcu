#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

static float parse_env_float(const char *name, float default_value) {
  const char *env = getenv(name);
  if (!env || env[0] == '\0') return default_value;
  char *endptr = NULL;
  float value = strtof(env, &endptr);
  if (endptr == env || (endptr && *endptr != '\0')) return default_value;
  return value;
}

static int64_t parse_env_i64(const char *name, int64_t default_value) {
  const char *env = getenv(name);
  if (!env || env[0] == '\0') return default_value;
  char *endptr = NULL;
  long long value = strtoll(env, &endptr, 10);
  if (endptr == env || (endptr && *endptr != '\0')) return default_value;
  return (int64_t)value;
}

static int index_to_ij(int64_t index, int64_t M, int64_t N, int64_t *out_i,
                       int64_t *out_j) {
  if (index < 0 || M <= 0 || N <= 0) return 0;
  int64_t i = index / N;
  int64_t j = index % N;
  if (i < 0 || i >= M || j < 0 || j >= N) return 0;
  *out_i = i;
  *out_j = j;
  return 1;
}

static int is_env_value(const char *env, const char *expected) {
  if (!env || !expected) return 0;
  while (*env && *expected) {
    if (tolower((unsigned char)*env) != tolower((unsigned char)*expected))
      return 0;
    ++env;
    ++expected;
  }
  return *env == '\0' && *expected == '\0';
}

static void apply_trivial_pattern(float *C, int64_t M, int64_t N, int64_t i,
                                  int64_t j, float delta) {
  if (M <= 0 || N <= 0) return;
  // For degenerate shapes, keep the total sum invariant.
  if (M == 1 && N == 1) return;
  if (M == 1) {
    if (j >= N - 1) j = N - 2;
    C[j] -= delta;
    C[j + 1] += delta;
    return;
  }
  if (N == 1) {
    if (i >= M - 1) i = M - 2;
    C[i * N] -= delta;
    C[(i + 1) * N] += delta;
    return;
  }
  if (i >= M - 1) i = M - 2;
  if (j >= N - 1) j = N - 2;
  // 2x2 balanced pattern:
  //  - +
  //  + -
  C[i * N + j] -= delta;
  C[i * N + (j + 1)] += delta;
  C[(i + 1) * N + j] += delta;
  C[(i + 1) * N + (j + 1)] -= delta;
}

static void apply_checkered_pattern(float *C, int64_t M, int64_t N,
                                    float delta) {
  if (M <= 0 || N <= 0) return;
  if (M == 1 && N == 1) return;
  if (M == 1) {
    for (int64_t j = 0; j + 1 < N; j += 2) {
      C[j] -= delta;
      C[j + 1] += delta;
    }
    return;
  }
  if (N == 1) {
    for (int64_t i = 0; i + 1 < M; i += 2) {
      C[i * N] -= delta;
      C[(i + 1) * N] += delta;
    }
    return;
  }
  for (int64_t i = 0; i + 1 < M; i += 2) {
    for (int64_t j = 0; j + 1 < N; j += 2) {
      apply_trivial_pattern(C, M, N, i, j, delta);
    }
  }
}

// In-place fault injection on C[M,N]. Injection is strictly gated by the
// per-call enable_flag inserted by the FI compiler pass.
void fi_plugin_f32(float *C, int64_t M, int64_t N, int64_t enable_flag) {
  if (!C || M <= 0 || N <= 0) return;

  if (!enable_flag) return;

  const char *pattern_env = getenv("IREE_FI_PATTERN");
  float delta = parse_env_float("IREE_FI_VALUE", 1.0f);
  int64_t index = parse_env_i64("IREE_FI_INDEX", 0);

  int64_t i = 0;
  int64_t j = 0;
  if (!index_to_ij(index, M, N, &i, &j)) {
    i = 0;
    j = 0;
  }
  if (is_env_value(pattern_env, "trivial")) {
    apply_trivial_pattern(C, M, N, i, j, delta);
    return;
  }
  if (is_env_value(pattern_env, "checkered")) {
    apply_checkered_pattern(C, M, N, delta);
    return;
  }
  C[i * N + j] += delta;
}

#ifdef __cplusplus
}  // extern "C"
#endif
