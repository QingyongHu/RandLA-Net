@echo off

REM Compile cpp subsampling
cd cpp_subsampling
python setup.py build_ext --inplace
cd ..
