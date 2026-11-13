// hq_vec.hpp -- tiny named-array container for shared test vectors.
// File = "HQV1" magic, then records: u32 name_len, name bytes, u32 dtype, u64 count, raw data.
// dtype: 0=u8 1=i8 2=u16 3=i32 4=u32 5=f32 6=u64 7=f64.  Little-endian only.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace hqv {

struct Arr { uint32_t dtype = 0; std::vector<uint8_t> bytes; };
inline size_t esz(uint32_t dt) { static const size_t s[] = {1, 1, 2, 4, 4, 4, 8, 8}; return s[dt]; }

template <class T> struct DT;
template <> struct DT<uint8_t>  { static constexpr uint32_t v = 0; };
template <> struct DT<int8_t>   { static constexpr uint32_t v = 1; };
template <> struct DT<uint16_t> { static constexpr uint32_t v = 2; };
template <> struct DT<int32_t>  { static constexpr uint32_t v = 3; };
template <> struct DT<uint32_t> { static constexpr uint32_t v = 4; };
template <> struct DT<float>    { static constexpr uint32_t v = 5; };
template <> struct DT<uint64_t> { static constexpr uint32_t v = 6; };
template <> struct DT<double>   { static constexpr uint32_t v = 7; };

struct File {
  std::map<std::string, Arr> a;
  std::vector<std::string> order;

  template <class T> void put(const std::string& n, const std::vector<T>& v) {
    Arr r; r.dtype = DT<T>::v; r.bytes.resize(v.size() * sizeof(T));
    if (!v.empty()) memcpy(r.bytes.data(), v.data(), r.bytes.size());
    if (!a.count(n)) order.push_back(n);
    a[n] = std::move(r);
  }
  template <class T> void put1(const std::string& n, T x) { put<T>(n, std::vector<T>{x}); }
  bool has(const std::string& n) const { return a.count(n) != 0; }
  template <class T> std::vector<T> get(const std::string& n) const {
    auto it = a.find(n);
    if (it == a.end()) throw std::runtime_error("missing array " + n);
    if (it->second.dtype != DT<T>::v) throw std::runtime_error("dtype mismatch " + n);
    std::vector<T> v(it->second.bytes.size() / sizeof(T));
    if (!v.empty()) memcpy(v.data(), it->second.bytes.data(), it->second.bytes.size());
    return v;
  }
  template <class T> T get1(const std::string& n) const { return get<T>(n).at(0); }

  void save(const std::string& path) const {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("open " + path);
    fwrite("HQV1", 1, 4, f);
    for (auto& n : order) {
      const Arr& r = a.at(n);
      uint32_t nl = (uint32_t)n.size(); fwrite(&nl, 4, 1, f); fwrite(n.data(), 1, nl, f);
      fwrite(&r.dtype, 4, 1, f); uint64_t c = r.bytes.size() / esz(r.dtype); fwrite(&c, 8, 1, f);
      if (!r.bytes.empty()) fwrite(r.bytes.data(), 1, r.bytes.size(), f);
    }
    fclose(f);
  }
  static File load(const std::string& path) {
    File F; FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("open " + path);
    char m[4]; if (fread(m, 1, 4, f) != 4 || memcmp(m, "HQV1", 4)) throw std::runtime_error("bad magic");
    for (;;) {
      uint32_t nl; if (fread(&nl, 4, 1, f) != 1) break;
      std::string n(nl, '\0'); if (fread(&n[0], 1, nl, f) != nl) throw std::runtime_error("trunc");
      Arr r; uint64_t c; if (fread(&r.dtype, 4, 1, f) != 1 || fread(&c, 8, 1, f) != 1) throw std::runtime_error("trunc");
      r.bytes.resize(c * esz(r.dtype));
      if (c && fread(r.bytes.data(), 1, r.bytes.size(), f) != r.bytes.size()) throw std::runtime_error("trunc");
      F.order.push_back(n); F.a[n] = std::move(r);
    }
    fclose(f); return F;
  }
};

}  // namespace hqv
