@echo off

pushd %~dp0

if "%1" == "release" (
    set Definitions=-O3
) else (
    set Definitions=-O0
)

IF NOT EXIST build mkdir build
pushd build 
clang++ -std=c++17 -march=native %Definitions% -g -D DEBUG -ferror-limit=5 -ffp-contract=off -o "convert.exe" "../src/rt/cmd/convert.cpp"
popd build

popd
