#pragma once
#include <cpuid.h>
#include <immintrin.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <setjmp.h>
#include <signal.h>
#include <vector>
#include "board.h"

namespace ataxx {

constexpr int W = 8, PS = 56;
typedef float V __attribute__((vector_size(32)));
typedef int VI __attribute__((vector_size(32)));

inline int pad(int n) { return (n + W - 1) / W * W; }
inline float* buffer(int n) { return (float*)aligned_alloc(64, (sizeof(float) * pad(n) + 63) / 64 * 64); }
inline V load(const float* p) { return *(const V*)p; }
inline void store(float* p, V v) { *(V*)p = v; }
inline V splat(float x) { return x - V{}; }
inline float total(V v) {
  float s = 0;
  for (int k = 0; k < W; k++) s += v[k];
  return s;
}

inline __m256 exp8(__m256 x) {
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.f)), _mm256_set1_ps(60.f));
  __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  x = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693147181f), x);
  __m256 y = _mm256_set1_ps(8.3334519073e-3f);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.f));
  return _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(y), _mm256_slli_epi32(_mm256_cvtps_epi32(n), 23)));
}

inline __m256 gelu8(__m256 x) {
  __m256 a = _mm256_mul_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), x), _mm256_set1_ps(0.70710678f));
  __m256 d = _mm256_fmadd_ps(a, _mm256_set1_ps(0.3275911f), _mm256_set1_ps(1.f)), t = _mm256_rcp_ps(d);
  t = _mm256_mul_ps(t, _mm256_fnmadd_ps(d, t, _mm256_set1_ps(2.f)));
  __m256 p = _mm256_fmadd_ps(_mm256_set1_ps(1.061405429f), t, _mm256_set1_ps(-1.453152027f));
  p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.421413741f));
  p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-0.284496736f));
  p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(0.254829592f));
  __m256 erf = _mm256_fnmadd_ps(_mm256_mul_ps(p, t), exp8(_mm256_mul_ps(_mm256_mul_ps(a, a), _mm256_set1_ps(-1.f))), _mm256_set1_ps(1.f));
  erf = _mm256_or_ps(erf, _mm256_and_ps(x, _mm256_set1_ps(-0.f)));
  return _mm256_mul_ps(_mm256_mul_ps(x, _mm256_set1_ps(0.5f)), _mm256_add_ps(erf, _mm256_set1_ps(1.f)));
}

inline void transpose8(__m256* r) {
  __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]), t1 = _mm256_unpackhi_ps(r[0], r[1]), t2 = _mm256_unpacklo_ps(r[2], r[3]), t3 = _mm256_unpackhi_ps(r[2], r[3]);
  __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]), t5 = _mm256_unpackhi_ps(r[4], r[5]), t6 = _mm256_unpacklo_ps(r[6], r[7]), t7 = _mm256_unpackhi_ps(r[6], r[7]);
  __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44), s1 = _mm256_shuffle_ps(t0, t2, 0xEE), s2 = _mm256_shuffle_ps(t1, t3, 0x44), s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
  __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44), s5 = _mm256_shuffle_ps(t4, t6, 0xEE), s6 = _mm256_shuffle_ps(t5, t7, 0x44), s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
  r[0] = _mm256_permute2f128_ps(s0, s4, 0x20), r[1] = _mm256_permute2f128_ps(s1, s5, 0x20), r[2] = _mm256_permute2f128_ps(s2, s6, 0x20), r[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
  r[4] = _mm256_permute2f128_ps(s0, s4, 0x31), r[5] = _mm256_permute2f128_ps(s1, s5, 0x31), r[6] = _mm256_permute2f128_ps(s2, s6, 0x31), r[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}

struct Dense {
  int in = 0, out = 0, wide = 0;
  float *w = nullptr, *b = nullptr;

  void read(FILE* f, int inputs, int outputs, const int* order = nullptr) {
    in = inputs, out = outputs, wide = pad(out);
    std::vector<float> raw((size_t)in * out);
    w = buffer(in * wide), b = buffer(wide);
    std::fill(w, w + in * wide, 0.f);
    std::fill(b, b + wide, 0.f);
    if (fread(raw.data(), 4, raw.size(), f) != raw.size() || fread(b, 4, out, f) != (size_t)out) abort();
    for (int o = 0; o < out; o++)
      for (int i = 0; i < in; i++) w[(order ? order[i] : i) * wide + o] = raw[(size_t)o * in + i];
  }

  template <int K>
  void one(const float* x, int o, float* y) const {
    V a[K];
    for (int k = 0; k < K; k++) a[k] = load(b + o + W * k);
    const float* p = w + o;
    for (int i = 0; i < in; i++, p += wide) {
      V u = splat(x[i]);
      for (int k = 0; k < K; k++) a[k] += u * load(p + W * k);
    }
    for (int k = 0; k < K; k++) store(y + o + W * k, a[k]);
  }

  template <int K>
  void two(const float* x0, const float* x1, int o, float* y0, float* y1) const {
    V a[K], c[K];
    for (int k = 0; k < K; k++) a[k] = c[k] = load(b + o + W * k);
    const float* p = w + o;
    for (int i = 0; i < in; i++, p += wide) {
      V u = splat(x0[i]), v = splat(x1[i]);
      for (int k = 0; k < K; k++) {
        V t = load(p + W * k);
        a[k] += u * t;
        c[k] += v * t;
      }
    }
    for (int k = 0; k < K; k++) store(y0 + o + W * k, a[k]), store(y1 + o + W * k, c[k]);
  }

  void run(const float* x, int stride, int n, float* y) const {
    int r = 0;
    for (; r + 2 <= n; r += 2, x += 2 * stride, y += 2 * wide) {
      int o = 0;
      for (; o + 4 * W <= wide; o += 4 * W) two<4>(x, x + stride, o, y, y + wide);
      for (; o < wide; o += W) two<1>(x, x + stride, o, y, y + wide);
    }
    for (; r < n; r++, x += stride, y += wide) {
      int o = 0;
      for (; o + 8 * W <= wide; o += 8 * W) one<8>(x, o, y);
      for (; o < wide; o += W) one<1>(x, o, y);
    }
  }
};

inline int VNNI = 0;
inline sigjmp_buf vnniJump;

inline void vnniSignal(int) { siglongjmp(vnniJump, 1); }

__attribute__((target("avx2,avx512f,avx512vl"), noinline)) inline int vnniInstruction() {
  __m256i a = _mm256_setzero_si256(), u = _mm256_set1_epi8(1), m = _mm256_set1_epi8(2);
  __asm__ volatile("%{evex%} vpdpbusd %2, %1, %0" : "+x"(a) : "x"(u), "x"(m));
  return _mm256_extract_epi32(a, 0);
}

inline bool vnniProbe() {
  struct sigaction action{}, previous;
  action.sa_handler = vnniSignal;
  sigemptyset(&action.sa_mask);
  sigaction(SIGILL, &action, &previous);
  volatile bool supported = false;
  if (!sigsetjmp(vnniJump, 1)) supported = vnniInstruction() == 8;
  sigaction(SIGILL, &previous, nullptr);
  return supported;
}

inline int vnni() {
  static const int level = []() -> int {
    unsigned a, b, c, d;
    __cpuid(1, a, b, c, d);
    if (!(c >> 27 & 1)) return 0;
    unsigned lo, hi;
    __asm__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    __cpuid_count(7, 0, a, b, c, d);
    if ((lo & 0xe6) == 0xe6 && ((c >> 11 & 1) || vnniProbe())) return 2;
    __cpuid_count(7, 1, a, b, c, d);
    return a >> 4 & 1;
  }();
  return level;
}

struct Dense8 {
  int in = 0, out = 0, wide = 0, deep = 0;
  int8_t* w = nullptr;
  uint8_t* q = nullptr;
  float *mult = nullptr, *b = nullptr, lo = 0, inv = 0;

  void read(FILE* f, int inputs, int outputs) {
    in = inputs, out = outputs, wide = (out + 15) / 16 * 16, deep = (in + 31) / 32 * 32;
    float range[2];
    w = (int8_t*)aligned_alloc(64, (size_t)wide * deep);
    q = (uint8_t*)aligned_alloc(64, (size_t)(SQ + 1) * deep);
    mult = buffer(wide), b = buffer(wide);
    std::fill(w, w + wide * deep, 0);
    std::fill(q, q + (SQ + 1) * deep, 0);
    std::fill(mult, mult + wide, 0.f);
    std::fill(b, b + wide, 0.f);
    if (fread(range, 4, 2, f) != 2) abort();
    lo = range[0], inv = 1 / range[1];
    std::vector<int8_t> raw((size_t)out * in);
    if (fread(raw.data(), 1, raw.size(), f) != raw.size()) abort();
    for (int o = 0; o < out; o++)
      for (int i = 0; i < in; i++) w[((i / 4) * wide + o) * 4 + i % 4] = raw[(size_t)o * in + i];
    if (fread(mult, 4, out, f) != (size_t)out || fread(b, 4, out, f) != (size_t)out) abort();
  }

#define ATAXX_PRODUCT(NAME, TARGET, STEP) \
  __attribute__((target(TARGET))) void NAME(int n, float* y) const { \
    const __m256i ones = _mm256_set1_epi16(1), twos = _mm256_set1_epi16(2), half = _mm256_set1_epi8(127), bit = _mm256_set1_epi8(1); \
    (void)ones, (void)twos, (void)half, (void)bit; \
    for (int r = 0; r < n; r += 4) { \
      const uint8_t* x[4]; \
      float* yr[4]; \
      for (int e = 0; e < 4; e++) x[e] = q + std::min(r + e, n - 1) * deep, yr[e] = y + std::min(r + e, n - 1) * wide; \
      for (int o = 0; o < wide; o += 16) { \
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0, c0 = a0, c1 = a0, c2 = a0, c3 = a0; \
        const int8_t* p = w + o * 4; \
        for (int k = 0; k < deep; k += 4, p += wide * 4) { \
          __m256i m0 = _mm256_load_si256((const __m256i*)p), m1 = _mm256_load_si256((const __m256i*)(p + 32)); \
          __m256i u0 = _mm256_set1_epi32(*(const int*)(x[0] + k)), u1 = _mm256_set1_epi32(*(const int*)(x[1] + k)); \
          __m256i u2 = _mm256_set1_epi32(*(const int*)(x[2] + k)), u3 = _mm256_set1_epi32(*(const int*)(x[3] + k)); \
          a0 = STEP(a0, u0, m0), c0 = STEP(c0, u0, m1), a1 = STEP(a1, u1, m0), c1 = STEP(c1, u1, m1); \
          a2 = STEP(a2, u2, m0), c2 = STEP(c2, u2, m1), a3 = STEP(a3, u3, m0), c3 = STEP(c3, u3, m1); \
        } \
        __m256 m = _mm256_load_ps(mult + o), d = _mm256_load_ps(b + o), m8 = _mm256_load_ps(mult + o + 8), d8 = _mm256_load_ps(b + o + 8); \
        _mm256_store_ps(yr[3] + o, _mm256_fmadd_ps(_mm256_cvtepi32_ps(a3), m, d)), _mm256_store_ps(yr[3] + o + 8, _mm256_fmadd_ps(_mm256_cvtepi32_ps(c3), m8, d8)); \
        _mm256_store_ps(yr[2] + o, _mm256_fmadd_ps(_mm256_cvtepi32_ps(a2), m, d)), _mm256_store_ps(yr[2] + o + 8, _mm256_fmadd_ps(_mm256_cvtepi32_ps(c2), m8, d8)); \
        _mm256_store_ps(yr[1] + o, _mm256_fmadd_ps(_mm256_cvtepi32_ps(a1), m, d)), _mm256_store_ps(yr[1] + o + 8, _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1), m8, d8)); \
        _mm256_store_ps(yr[0] + o, _mm256_fmadd_ps(_mm256_cvtepi32_ps(a0), m, d)), _mm256_store_ps(yr[0] + o + 8, _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0), m8, d8)); \
      } \
    } \
  }
#define ATAXX_MADD(a, u, m) \
  _mm256_add_epi32(a, _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(u, 1), half), m), twos), \
                                      _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(u, bit), m), ones)))
#define ATAXX_AVXVNNI(a, u, m) _mm256_dpbusd_avx_epi32(a, u, m)
  ATAXX_PRODUCT(product, "avx2", ATAXX_MADD)
  ATAXX_PRODUCT(productA, "avx2,avxvnni", ATAXX_AVXVNNI)

  __attribute__((target("avx2,avx512f,avx512vl,avx512vnni"))) void productT(int n, float* y) const {
    for (int r = 0; r < n; r += 7) {
      const uint8_t* x[7];
      float* yr[7];
      for (int e = 0; e < 7; e++) x[e] = q + std::min(r + e, n - 1) * deep, yr[e] = y + std::min(r + e, n - 1) * wide;
      for (int o = 0; o < wide; o += 16) {
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, c0 = a0, c1 = a0, c2 = a0, c3 = a0, c4 = a0, c5 = a0, c6 = a0;
        const int8_t* p = w + o * 4;
        for (int k = 0; k < deep; k += 4, p += wide * 4) {
          __m256i m0 = _mm256_load_si256((const __m256i*)p), m1 = _mm256_load_si256((const __m256i*)(p + 32)), u;
#define ATAXX_TILE(A, C, E) u = _mm256_set1_epi32(*(const int*)(x[E] + k)); __asm__("%{evex%} vpdpbusd %2, %1, %0" : "+v"(A) : "v"(u), "v"(m0)); __asm__("%{evex%} vpdpbusd %2, %1, %0" : "+v"(C) : "v"(u), "v"(m1));
          ATAXX_TILE(a0, c0, 0) ATAXX_TILE(a1, c1, 1) ATAXX_TILE(a2, c2, 2) ATAXX_TILE(a3, c3, 3) ATAXX_TILE(a4, c4, 4) ATAXX_TILE(a5, c5, 5) ATAXX_TILE(a6, c6, 6)
        }
        __m256 m = _mm256_load_ps(mult + o), d = _mm256_load_ps(b + o), m8 = _mm256_load_ps(mult + o + 8), d8 = _mm256_load_ps(b + o + 8);
#define ATAXX_OUT(A, C, E) _mm256_store_ps(yr[E] + o, _mm256_fmadd_ps(_mm256_cvtepi32_ps(A), m, d)), _mm256_store_ps(yr[E] + o + 8, _mm256_fmadd_ps(_mm256_cvtepi32_ps(C), m8, d8));
        ATAXX_OUT(a6, c6, 6) ATAXX_OUT(a5, c5, 5) ATAXX_OUT(a4, c4, 4) ATAXX_OUT(a3, c3, 3) ATAXX_OUT(a2, c2, 2) ATAXX_OUT(a1, c1, 1) ATAXX_OUT(a0, c0, 0)
      }
    }
  }

  void run(const float* x, int stride, int n, float* y) const {
    const __m256 base = _mm256_set1_ps(lo), scale = _mm256_set1_ps(inv);
    const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    uint8_t* __restrict__ qq = q;
    auto cvt = [&](const float* p) { return _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sub_ps(_mm256_load_ps(p), base), scale)); };
    for (int r = 0; r < n; r++) {
      const float* row = x + r * stride;
      for (int i = 0; i < in; i += 32) {
        __m256i ab = _mm256_packs_epi32(cvt(row + i), cvt(row + i + 8)), cd = _mm256_packs_epi32(cvt(row + i + 16), cvt(row + i + 24));
        _mm256_store_si256((__m256i*)(qq + r * deep + i), _mm256_permutevar8x32_epi32(_mm256_packus_epi16(ab, cd), order));
      }
    }
    if (VNNI == 2) productT(n, y);
    else if (VNNI == 1) productA(n, y);
    else product(n, y);
  }
};

struct Norm {
  float *w, *b;
  int c;

  void read(FILE* f, int channels) {
    c = channels;
    w = buffer(c), b = buffer(c);
    if (fread(w, 4, c, f) != (size_t)c || fread(b, 4, c, f) != (size_t)c) abort();
  }

  void run(const float* x, float* y) const {
    for (int r = 0; r < SQ; r++, x += c, y += c) {
      V s = splat(0), q = splat(0);
      for (int i = 0; i < c; i += W) {
        V v = load(x + i);
        s += v;
        q += v * v;
      }
      float m = total(s) / c, scale = 1 / std::sqrt(total(q) / c - m * m + 1e-5f);
      for (int i = 0; i < c; i += W) store(y + i, (load(x + i) - m) * scale * load(w + i) + load(b + i));
    }
  }
};

struct Layer {
  Norm norm1, norm2;
  Dense qkv, proj, up, down;
  Dense8 qkv8, proj8, up8, down8;
  float* bias;
};

struct Transformer {
  int blocks, c, heads, ratio, smooth, quant, key, d, soft, span;
  Dense stem, policy1, pool, policy2, value1, value2, value3;
  Norm norm;
  std::vector<Layer> layers;
  float *position, *x, *y, *big, *att, *kt, *score;

  explicit Transformer(const char* path) {
    FILE* f = fopen(path, "rb");
    int head[9];
    if (!f || fread(head, 4, 9, f) != 9) abort();
    blocks = head[0], c = head[1], heads = head[2], ratio = head[3], smooth = head[4], quant = head[5], soft = head[8];
    VNNI = getenv("ATAXX_NOVNNI") ? 0 : vnni();
    key = head[6] ? head[6] : c / heads, d = head[7] ? head[7] : c / heads, span = pad(heads * (2 * key + d));
    int order[PLANES * 9];
    for (int p = 0; p < PLANES; p++)
      for (int k = 0; k < 9; k++) order[p * 9 + k] = k * PLANES + p;
    stem.read(f, PLANES * 9, c, order);
    position = buffer(SQ * c);
    if (fread(position, 4, SQ * c, f) != (size_t)SQ * c) abort();
    layers.resize(blocks);
    for (Layer& l : layers) {
      l.norm1.read(f, c);
      if (quant) l.qkv8.read(f, c, heads * (2 * key + d)), l.proj8.read(f, heads * d, c);
      else l.qkv.read(f, c, heads * (2 * key + d)), l.proj.read(f, heads * d, c);
      l.bias = buffer(heads * SQ * PS);
      for (int i = 0; i < heads * SQ; i++) {
        std::fill(l.bias + i * PS, l.bias + (i + 1) * PS, -1e30f);
        if (fread(l.bias + i * PS, 4, SQ, f) != SQ) abort();
      }
      l.norm2.read(f, c);
      if (quant) l.up8.read(f, c, ratio * c), l.down8.read(f, ratio * c, c);
      else l.up.read(f, c, ratio * c), l.down.read(f, ratio * c, c);
    }
    norm.read(f, c);
    policy1.read(f, c, 32);
    pool.read(f, 2 * c, 32);
    policy2.read(f, 32, KINDS);
    value1.read(f, c, 32);
    value2.read(f, 64, 64);
    value3.read(f, 64, 4);
    fclose(f);
    x = buffer(SQ * c), y = buffer(SQ * c), big = buffer(SQ * std::max(span, ratio * c)), att = buffer(SQ * c);
    kt = buffer(key * PS), score = buffer(2 * PS);
  }

  static float total8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)));
  }

  template <int L>
  void mix(const float* p, float* y0, float* y1, float inv0, float inv1) const {
    __m256 a0[L], a1[L], c0[L], c1[L];
    for (int e = 0; e < L; e++) a0[e] = a1[e] = c0[e] = c1[e] = _mm256_setzero_ps();
    for (int j = 0; j < SQ - 1; j += 2, p += 2 * span) {
      __m256 w0 = _mm256_broadcast_ss(score + j), w1 = _mm256_broadcast_ss(score + PS + j);
      __m256 x0 = _mm256_broadcast_ss(score + j + 1), x1 = _mm256_broadcast_ss(score + PS + j + 1);
      for (int e = 0; e < L; e++) {
        __m256 vv = _mm256_load_ps(p + W * e), vx = _mm256_load_ps(p + span + W * e);
        a0[e] = _mm256_fmadd_ps(w0, vv, a0[e]);
        a1[e] = _mm256_fmadd_ps(w1, vv, a1[e]);
        c0[e] = _mm256_fmadd_ps(x0, vx, c0[e]);
        c1[e] = _mm256_fmadd_ps(x1, vx, c1[e]);
      }
    }
    __m256 w0 = _mm256_broadcast_ss(score + SQ - 1), w1 = _mm256_broadcast_ss(score + PS + SQ - 1);
    for (int e = 0; e < L; e++) {
      __m256 vv = _mm256_load_ps(p + W * e);
      _mm256_store_ps(y1 + W * e, _mm256_mul_ps(_mm256_add_ps(_mm256_fmadd_ps(w1, vv, a1[e]), c1[e]), _mm256_set1_ps(inv1)));
      _mm256_store_ps(y0 + W * e, _mm256_mul_ps(_mm256_add_ps(_mm256_fmadd_ps(w0, vv, a0[e]), c0[e]), _mm256_set1_ps(inv0)));
    }
  }

  void attend(const Layer& l) {
    float scale = 1 / std::sqrt((float)key);
    const int hv = heads * d;
    for (int h = 0; h < heads; h++) {
      const float *q = big + h * key, *k = big + (heads + h) * key, *v = big + 2 * heads * key + h * d;
      for (int t = 0; t < key; t += W)
        for (int j = 0; j < PS; j += W) {
          __m256 r[W];
          for (int e = 0; e < W; e++) r[e] = j + e < SQ ? _mm256_mul_ps(_mm256_load_ps(k + (j + e) * span + t), _mm256_set1_ps(scale)) : _mm256_setzero_ps();
          transpose8(r);
          for (int e = 0; e < W; e++) _mm256_store_ps(kt + (t + e) * PS + j, r[e]);
        }
      for (int i = 0; i < SQ; i += 2) {
        int i1 = std::min(i + 1, SQ - 1);
        const float *b0 = l.bias + (h * SQ + i) * PS, *b1 = l.bias + (h * SQ + i1) * PS, *q0 = q + i * span, *q1 = q + i1 * span;
        __m256 s0[PS / W], s1[PS / W];
        for (int j = 0; j < PS / W; j++) s0[j] = _mm256_load_ps(b0 + W * j), s1[j] = _mm256_load_ps(b1 + W * j);
        for (int t = 0; t < key; t++) {
          __m256 u0 = _mm256_broadcast_ss(q0 + t), u1 = _mm256_broadcast_ss(q1 + t);
          for (int j = 0; j < PS / W; j++) {
            __m256 kv = _mm256_load_ps(kt + t * PS + W * j);
            s0[j] = _mm256_fmadd_ps(u0, kv, s0[j]);
            s1[j] = _mm256_fmadd_ps(u1, kv, s1[j]);
          }
        }
        float inv0 = 1.f / SQ, inv1 = inv0;
        if (soft) {
          __m256 t0 = _mm256_setzero_ps(), t1 = t0;
          for (int j = 0; j < PS / W; j++) {
            s0[j] = exp8(s0[j]), s1[j] = exp8(s1[j]);
            t0 = _mm256_add_ps(t0, s0[j]), t1 = _mm256_add_ps(t1, s1[j]);
          }
          inv0 = 1 / total8(t0), inv1 = 1 / total8(t1);
        } else {
          for (int j = 0; j < PS / W; j++) s0[j] = _mm256_max_ps(s0[j], _mm256_setzero_ps()), s1[j] = _mm256_max_ps(s1[j], _mm256_setzero_ps());
        }
        for (int j = 0; j < PS / W; j++) _mm256_store_ps(score + W * j, s0[j]), _mm256_store_ps(score + PS + W * j, s1[j]);
        if (d == 8) mix<1>(v, att + i * hv + h * d, att + i1 * hv + h * d, inv0, inv1);
        else if (d == 16) mix<2>(v, att + i * hv + h * d, att + i1 * hv + h * d, inv0, inv1);
        else mix<3>(v, att + i * hv + h * d, att + i1 * hv + h * d, inv0, inv1);
      }
    }
  }

  void eval(const float* feat, float* policy, float* value) {
    float patch[SQ * PLANES * 9];
    for (int s = 0; s < SQ; s++)
      for (int k = 0; k < 9; k++) {
        int px = s % 7 + k % 3 - 1, py = s / 7 + k / 3 - 1;
        for (int p = 0; p < PLANES; p++) patch[(s * 9 + k) * PLANES + p] = onBoard(px, py) ? feat[p * SQ + py * 7 + px] : 0.f;
      }
    stem.run(patch, PLANES * 9, SQ, x);
    for (int i = 0; i < SQ * c; i += W) store(x + i, load(x + i) + load(position + i));
    for (const Layer& l : layers) {
      l.norm1.run(x, y);
      if (quant) l.qkv8.run(y, c, SQ, big);
      else l.qkv.run(y, c, SQ, big);
      attend(l);
      if (quant) l.proj8.run(att, heads * d, SQ, y);
      else l.proj.run(att, heads * d, SQ, y);
      for (int i = 0; i < SQ * c; i += W) store(x + i, load(x + i) + load(y + i));
      l.norm2.run(x, y);
      if (quant) l.up8.run(y, c, SQ, big);
      else l.up.run(y, c, SQ, big);
      for (int i = 0; i < SQ * ratio * c; i += W) _mm256_store_ps(big + i, smooth ? gelu8(_mm256_load_ps(big + i)) : _mm256_max_ps(_mm256_load_ps(big + i), _mm256_setzero_ps()));
      if (quant) l.down8.run(big, ratio * c, SQ, y);
      else l.down.run(big, ratio * c, SQ, y);
      for (int i = 0; i < SQ * c; i += W) store(x + i, load(x + i) + load(y + i));
    }
    norm.run(x, y);
    alignas(64) float pooled[2 * 256], hidden[SQ * 32], extra[32], out[SQ * 32], mid[64], last[16];
    summarize(y, c, c, pooled);
    policy1.run(y, c, SQ, hidden);
    pool.run(pooled, 0, 1, extra);
    for (int s = 0; s < SQ; s++)
      for (int j = 0; j < 32; j++) hidden[s * 32 + j] = std::max(0.f, hidden[s * 32 + j] + extra[j]);
    policy2.run(hidden, 32, SQ, out);
    for (int s = 0; s < SQ; s++)
      for (int k = 0; k < KINDS; k++) policy[k * SQ + s] = out[s * policy2.wide + k];
    value1.run(y, c, SQ, hidden);
    for (int i = 0; i < SQ * 32; i++) hidden[i] = std::max(0.f, hidden[i]);
    summarize(hidden, 32, 32, pooled);
    value2.run(pooled, 0, 1, mid);
    for (int i = 0; i < 64; i++) mid[i] = std::max(0.f, mid[i]);
    value3.run(mid, 0, 1, last);
    std::copy(last, last + 4, value);
  }

  static void summarize(const float* t, int stride, int n, float* out) {
    for (int j = 0; j < n; j++) {
      float sum = 0, mx = -1e30f;
      for (int s = 0; s < SQ; s++) sum += t[s * stride + j], mx = std::max(mx, t[s * stride + j]);
      out[j] = sum / SQ, out[n + j] = mx;
    }
  }
};


}
