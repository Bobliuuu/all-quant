# Adds an env-gated UQ_FA_GRID perf set to llama.cpp tests/test-backend-ops.cpp (Hq32 Hkv8 D128, mask on, B via ne3 = distinct KV per sequence)
import sys,re
p=sys.argv[1]; s=open(p,encoding='utf-8').read()
key="static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {\n    std::vector<std::unique_ptr<test_case>> test_cases;\n"
assert key in s
add=key+'''    if (getenv("UQ_FA_GRID")) {
        for (ggml_type t : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) if (strcmp(getenv("UQ_FA_GRID"), "all") == 0 || strcmp(getenv("UQ_FA_GRID"), ggml_type_name(t)) == 0)
        for (int64_t B : {1, 2, 8})
        for (int64_t kv : {4096, 16384, 32768})
        for (int64_t nb : {1, 2, 4, 8})
            test_cases.emplace_back(new test_flash_attn_ext(128, 128, 8, {4, B}, kv, nb, true, false, 0, 0, GGML_PREC_F32, t, t));
        return test_cases;
    }
'''
if 'UQ_FA_GRID' in s:
    s=re.sub(r'    if \(getenv\("UQ_FA_GRID"\)\) \{.*?return test_cases;\n    \}\n','',s,flags=re.S)
s=s.replace(key,add,1)
open(p,'w',encoding='utf-8',newline='\n').write(s)
