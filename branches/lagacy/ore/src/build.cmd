@echo off
rem set CUDA_SDK_ROOT=C:\Program Files\NVIDIA Corporation\NVIDIA CUDA SDK

set cfg=release

bjam toolset=msvc-9.0 %cfg%
mkdir %cfg%
copy bin\msvc-9.0\%cfg%\threading-multi\_ore.pyd ..\
