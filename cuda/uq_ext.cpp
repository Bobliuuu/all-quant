// PyTorch binding for the UltraQuant CUDA kernel.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "uq_decode.h"

torch::Tensor uq_decode_cuda(torch::Tensor q, torch::Tensor kc, torch::Tensor ks, torch::Tensor vc, torch::Tensor vs,
                             torch::Tensor bt, torch::Tensor sl, int64_t nsplit, int64_t nwarps, int64_t stages, int64_t variant, int64_t mb,
                             double sm_scale, torch::optional<torch::Tensor> mo_, torch::optional<torch::Tensor> ml_,
                             torch::optional<torch::Tensor> out_, int64_t fmt, double k_ts, double v_ts) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.is_contiguous(), "q must be contiguous bf16 cuda");
  TORCH_CHECK(bt.scalar_type() == torch::kInt32 && sl.scalar_type() == torch::kInt32, "bt/sl must be int32");
  UQParams p{};
  p.B = q.size(0); p.K = q.size(1); p.HQ = q.size(2); p.D = q.size(3);
  p.HKV = kc.size(1); p.PAGE = kc.size(2); p.max_pages = bt.size(1); p.nsplit = (int)nsplit;
  p.sm_scale = (float)sm_scale; p.variant = (int)variant; p.mb = (int)mb; p.fmt = (int)fmt; p.k_ts = (float)k_ts; p.v_ts = (float)v_ts;
  p.q = q.data_ptr(); p.kc = kc.data_ptr<uint8_t>(); p.ks = ks.data_ptr<uint8_t>();
  p.vc = vc.data_ptr<uint8_t>(); p.vs = vs.data_ptr<uint8_t>();
  p.block_table = bt.data_ptr<int32_t>(); p.seq_lens = sl.data_ptr<int32_t>();
  auto f32 = q.options().dtype(torch::kFloat32);
  torch::Tensor mo = mo_.has_value() ? *mo_ : torch::empty({(long)uq_mo_elems(p)}, f32);
  torch::Tensor ml = ml_.has_value() ? *ml_ : torch::empty({(long)uq_ml_elems(p)}, f32);
  torch::Tensor out = out_.has_value() ? *out_ : torch::empty_like(q);
  TORCH_CHECK(mo.numel() >= (long)uq_mo_elems(p) && ml.numel() >= (long)uq_ml_elems(p), "scratch too small");
  p.mo = mo.data_ptr<float>(); p.ml = ml.data_ptr<float>(); p.out = out.data_ptr();
  cudaError_t e = uq_decode_launch(p, (int)nwarps, (int)stages, c10::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == cudaSuccess, "uq_decode_launch failed: ", cudaGetErrorString(e));
  return out;
}
int64_t uq_block_m_py(int64_t K, int64_t HQ, int64_t HKV) { return uq_block_m(K, HQ, HKV); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &uq_decode_cuda, "UltraQuant FP4 paged K-query decode (CUDA)");
  m.def("block_m", &uq_block_m_py);
  m.def("has_sm90", []() { return (int64_t)uq_has_sm90(); });
  m.def("workspace_floats", [](int64_t B, int64_t K, int64_t HQ, int64_t HKV, int64_t D, int64_t nsplit) {
    UQParams p{}; p.B = B; p.K = K; p.HQ = HQ; p.HKV = HKV; p.D = D; p.nsplit = (int)nsplit;
    return (int64_t)(uq_mo_elems(p) + uq_ml_elems(p)); });
}
