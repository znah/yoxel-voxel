call "%VS90COMNTOOLS%\vsvars32.bat"

del /q ..\_ptx\*.*
cd ..\_ptx
set CUDA_SDK_ROOT=C:\Program Files\NVIDIA Corporation\NVIDIA CUDA SDK
"%CUDA_BIN_PATH%\nvcc.exe"  --opencc-options="-LIST:source=on" -Xptxas -v  -c -DWIN32 -D_CONSOLE -D_MBCS -keep -Xcompiler /EHsc,/W3,/nologo,/O2,/Zi,/MT -I"%CUDA_INC_PATH%" -I"../cpp" -o trace.obj ..\cpp\trace.cu
rem *keep