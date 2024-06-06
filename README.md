# Installation

To build and compare the generated code use:

```
mkdir build_no_fix
pushd build_no_fix
cmake .. -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_BUILD_TYPE=Custom -DCMAKE_CXX_FLAGS="-g -mcpu=neoverse-v2 -Ofast -fPIC -DNDEBUG" -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_CUDA_FLAGS="-diag-suppress 177 -fPIC --save-temps --verbose --generate-line-info -Xptxas=-v --expt-relaxed-constexpr -DNDEBUG" -DGT_NEIGHBOR_FIX=OFF
cmake --build .
popd
mkdir build_with_fix
pushd build_with_fix
cmake .. -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_BUILD_TYPE=Custom -DCMAKE_CXX_FLAGS="-g -mcpu=neoverse-v2 -Ofast -fPIC -DNDEBUG" -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_CUDA_FLAGS="-diag-suppress 177 -fPIC --save-temps --verbose --generate-line-info -Xptxas=-v --expt-relaxed-constexpr -DNDEBUG" -DGT_NEIGHBOR_FIX=OFF
cmake --build .
popd
```

Compare the generated PTX files in:
```
build_no_fix/copy_neighbor_gtfn_bindings.ptx
```
and
```
build_with_fix/copy_neighbor_gtfn_bindings.ptx
```

Look at the number of `ld.global.s32` instructions in `build_no_fix/copy_neighbor_gtfn_bindings.ptx` (8, 1 for every `shift` call in the kernel) and `ld.global.u32` instructions in `build_with_fix/copy_neighbor_gtfn_bindings.ptx` (4, 1 for every neighbor access).

The kernel is defiend in
```
copy_neighbor_gtfn.hpp
```