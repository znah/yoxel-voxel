@echo off
rem set CUDA_SDK_ROOT=C:\Program Files\NVIDIA Corporation\NVIDIA CUDA SDK

set cfg=release

bjam toolset=msvc-7.1 %cfg%
mkdir %cfg%
copy bin\msvc-7.1\%cfg%\threading-multi\_ore.pyd ..\
