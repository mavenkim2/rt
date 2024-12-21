@echo off

pushd %~dp0

IF NOT EXIST build mkdir build
pushd build 
clang++ -std=c++17 -march=native -ffp-contract=off -o compile.exe "../src/rt/cmd/compile.cpp"
popd build

popd
