## cuda_h100: capacity / agent / RAG / needle
| workload | kv | out tok/s | TTFT p50 s | TTFT p99 s | ITL p50 ms | ITL p99 ms | KV tokens | acc |
| agentdistinct_bf16 | auto | 646 | 0.256 | 31.121 | 14.2 | 239.1 | 730641 | - |
| agentdistinct_fp8 | fp8 | 632 | 0.257 | 32.098 | 13.9 | 243.4 | 1461360 | - |
| agentdistinct_q3_bf16 | auto | 99 | 45.174 | 52.762 | 24.4 | 403.9 | 380528 | - |
| agentdistinct_q3_fp8 | fp8 | 105 | 27.694 | 44.075 | 277.4 | 425.7 | 758016 | - |
| agentshared_bf16 | auto | 1431 | 0.367 | 1.474 | 11.1 | 140.1 | 730641 | - |
| agentshared_fp8 | fp8 | 1412 | 0.320 | 1.531 | 11.1 | 122.8 | 1461360 | - |
| needle_bf16 | auto | - | - | - | 0.0 | 0.0 | 730641 | 0.225 |
| needle_fp8 | fp8 | - | - | - | 0.0 | 0.0 | 1461360 | 0.225 |
| rag_bf16 | auto | - | 5.258 | 6.779 | 0.0 | 0.0 | 730641 | 0.236 |
| rag_fp8 | fp8 | - | 5.669 | 7.046 | 0.0 | 0.0 | 1461360 | 0.234 |
| rag_q3_bf16 | auto | - | 8.168 | 8.785 | 0.0 | 0.0 | 380528 | 0.588 |
| rag_q3_fp8 | fp8 | - | 7.924 | 8.755 | 0.0 | 0.0 | 758016 | 0.581 |

## spec-decode (n-gram, code edit)
| cfg | kv | K | conc | out tok/s | ITL p50 ms | tokens/step | acc/draft |
| spec_c32_k0_bf16 | auto | 0 | 32 | 2709 | 9.76 | 1.00 | - |
| spec_c32_k0_fp8 | fp8 | 0 | 32 | 2969 | 9.28 | 1.00 | - |
| spec_c32_k2_bf16 | auto | 2 | 32 | 4430 | 3.92 | 3.58 | 2.58 |
| spec_c32_k2_fp8 | fp8 | 2 | 32 | 4842 | 3.66 | 3.80 | 2.80 |
| spec_c32_k4_bf16 | auto | 4 | 32 | 5088 | 2.71 | 5.19 | 4.19 |
| spec_c32_k4_fp8 | fp8 | 4 | 32 | 6339 | 2.53 | 5.81 | 4.81 |
| spec_c32_k8_bf16 | auto | 8 | 32 | 5274 | 2.08 | 7.26 | 6.26 |
| spec_c32_k8_fp8 | fp8 | 8 | 32 | 6076 | 2.00 | 7.83 | 6.83 |
| spec_c8_k0_bf16 | auto | 0 | 8 | 1019 | 7.19 | 1.00 | - |
| spec_c8_k0_fp8 | fp8 | 0 | 8 | 1029 | 7.12 | 1.00 | - |
| spec_c8_k2_bf16 | auto | 2 | 8 | 2181 | 2.79 | 3.57 | 2.57 |
| spec_c8_k2_fp8 | fp8 | 2 | 8 | 2211 | 2.83 | 3.80 | 2.80 |
| spec_c8_k4_bf16 | auto | 4 | 8 | 2774 | 1.88 | 5.19 | 4.19 |
| spec_c8_k4_fp8 | fp8 | 4 | 8 | 2951 | 1.89 | 5.66 | 4.66 |
| spec_c8_k8_bf16 | auto | 8 | 8 | 3111 | 1.22 | 7.25 | 6.25 |
| spec_c8_k8_fp8 | fp8 | 8 | 8 | 3605 | 1.23 | 8.24 | 7.24 |

## drift vs bf16 greedy (first divergence token index, same 24 prompts)
- fp8: identical full sequence 1/24; median first-divergence token 30; min 1; mean 62
