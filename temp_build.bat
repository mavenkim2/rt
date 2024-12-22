@echo off

pushd %~dp0

IF NOT EXIST build mkdir build
pushd build 
clang++ -std=c++17 -march=native -ffp-contract=off -o "convert.exe" "../src/rt/cmd/convert.cpp"
popd build

popd
