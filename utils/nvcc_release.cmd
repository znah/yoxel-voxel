"%CUDA_BIN_PATH%\nvcc.exe"  -c -DWIN32 -D_CONSOLE -D_MBCS -Xcompiler /EHsc,/W3,/nologo,/O2,/Zi,/MT -I"%CUDA_INC_PATH%" -I"../cpp" -o Release\%~n1.obj %1
