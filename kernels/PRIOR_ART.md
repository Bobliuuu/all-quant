# Prior art (checked 2026-11-13, not exhaustive)

- **Pearl Research Labs, PearlQuant** (blog, https://pearlresearch.ai/research/blog/pearlquant):
  E8 lattice KV-cache codec with a Blackwell CuTe DSL decode-attention kernel inside vLLM.
  No code or format specification is public; the `pearl-research-labs` GitHub organization
  has no repositories. Everything in this repo is an independent reconstruction.
- **NestQuant** (Savkin, Porat, Ordentlich, Polyanskiy, ICML 2025, arXiv:2502.09720):
  nested-lattice (Voronoi) E8 codes for weights/activations/KV with Hadamard rotation and
  multi-beta scaling. Our `voronoi` variant follows its codebook construction.
- **HyperQuant** (MIT, github.com/moonmath-ai/HyperQuant): E8 (2·E8) nearest-point quantizer
  with Rice entropy coding and CUDA encoder/decoder kernels. Variable rate. Used here only as a
  read-only comparison in the rate-distortion experiment (imported, not copied).
- **Lattice VQ with leader indexing**: the RE8 codebooks of 3GPP AMR-WB+ and EVS index E8 points
  by absolute leader + permutation + signs; our ball-codebook index uses the same decomposition
  (our own implementation, our own leader order by norm).
- **TurboQuant** (Zandieh et al., arXiv:2504.19874): random rotation + Lloyd-Max scalar codebook.
  Our `turboquant_mse` baseline is a re-implementation of its MSE variant, not the authors' code.
- **UltraQuant** (same user, ~/code/ultraquant-more): FP4 KV verify kernel on three vendors.
  We followed its structure (shared vectors, porting log, results parsed from files) but share no code.

We found no public CUDA, HIP or SYCL decode-attention kernel for a ball-truncated (probable
points) E8 codebook with on-chip leader decoding.
