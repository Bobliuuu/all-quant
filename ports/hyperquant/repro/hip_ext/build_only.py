import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rocm_ext
e = rocm_ext.build()
print("EXT OK", [n for n in dir(e) if n.startswith("rice")])
