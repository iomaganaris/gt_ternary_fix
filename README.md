# Installation

To build and compare the generated code use:

```
python -m venv venv_gtfn
source venv_gtfn/bin/activate
pip install -r requirements.txt
# `gridtools` also need `boost` which can be installed either via `spack` or the system package manager or loaded using modules
mkdir build_without_fix
pushd build_without_fix
# Below command will build the project using `gridtools` `master`
cmake .. -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_BUILD_TYPE=Custom -DCMAKE_CXX_FLAGS="-g -mcpu=neoverse-v2 -Ofast -fPIC -DNDEBUG" -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_CUDA_FLAGS="-diag-suppress 177 -fPIC --save-temps --verbose --generate-line-info -Xptxas=-v --expt-relaxed-constexpr -DNDEBUG" -DGT_NEIGHBOR_FIX=OFF
cmake --build .
popd
mkdir build_with_fix
pushd build_with_fix
# Below command will build the project using the branch of https://github.com/GridTools/gridtools/pull/1779
cmake .. -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_BUILD_TYPE=Custom -DCMAKE_CXX_FLAGS="-g -mcpu=neoverse-v2 -Ofast -fPIC -DNDEBUG" -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_CUDA_FLAGS="-diag-suppress 177 -fPIC --save-temps --verbose --generate-line-info -Xptxas=-v --expt-relaxed-constexpr -DNDEBUG" -DGT_NEIGHBOR_FIX=ON
cmake --build .
popd
```

Compare the generated PTX files in:
```
build_without_fix/nabla4_gtfn_bindings.ptx
```
and
```
build_with_fix/nabla4_gtfn_bindings.ptx
```

Look at the number of `ld.global.s32` instructions in `build_without_fix/nabla4_gtfn_bindings.ptx` (15(+1 `ld.global.u32`), 1 for every `shift` call in the kernel) and `ld.global.u32` instructions in `build_with_fix/nabla4_gtfn_bindings.ptx` (8, 1 for every neighbor access).

The kernel is defiend in
```
nabla4_gtfn.hpp
```

To execute the kernel there is a python script which instantiates a large enough grid to show the difference. To use it:
```
$ PYTHONPATH=$(pwd)/build_without_fix:$PYTHONPATH python3.11 run_nabla4.py --klevels 65 --repetitions 20
nabla4_benchmark_unstructured_gtfn_gpu median runtime: 0.0012216320037841798

$ PYTHONPATH=$(pwd)/build_with_fix:$PYTHONPATH python3.11 run_nabla4.py --klevels 65 --repetitions 20
nabla4_benchmark_unstructured_gtfn_gpu median runtime: 0.0009779359996318817
```
