# CHFEMGPU

[license-image]: https://img.shields.io/badge/license-MIT-green.svg?style=flat
[license]: https://gitlab.com/cortezpedro/chfemgpu/-/blob/master/LICENSE

## Introduction

`chfem_gpu` is a software written in C and CUDA C for the computational homogenization of material samples characterized via $\mu$CT. As it is, the effective properties that can be evaluated are:

+ Thermal conductivity
+ Linear elasticity
+ Permeability

The program follows a matrix-free approach to image-based finite element analysis, exploring GPU resources with CUDA to achieve significant performance gains.

Memory efficiency is the north of this project. Our end goal is to be able to run large problems ($10^9$ DOFs) with relatively acessible graphics cards.

## Matrix-free strategies

+ Node-by-node
+ Element-by-element

## Considerations

+ Periodic boundary conditions
+ Linear shape functions for finite elements
+ Numerical stabilization for stokes flow

## Compile

```bash
~root$ cd compile
~root/compile$ python3 compile_chfem_gpu.py <options>
```

## Run

+ Linux

```bash
~root$ ./chfem_gpu [NEUTRAL_FILE] [RAW_IMAGE_FILE] <options>
```

+ Windows

```bash
~root$ chfem_gpu.exe [NEUTRAL_FILE] [RAW_IMAGE_FILE] <options>
```

## Test

```bash
~root$ cd test
~root/test$ python3 compile_test.py <options>
~root/test$ ./test
```
