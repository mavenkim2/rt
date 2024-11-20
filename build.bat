@echo off

set SSE2=-D __SSE__ -D __SSE2__
set SSE42=%SSE2% -D __SSE3__ -D __SSSE3__ -D __SSE4_1__ -D __SSE4_2__
set AVX=%SSE42% -arch:AVX
set AVX2=%SSE42% -arch:AVX2

set Dependencies=W:\rt\src\third_party\openvdb\nanovdb
set Definitions=-D NOMINMAX

if "%1" == "release" (
    call :AddReleaseFlags
) else (
    if "%2" == "release" (
        call :AddReleaseFlags %1
    ) else (
        call :AddDebugFlags %1
    )
) 
if "%1" == "reldebug" (
    call :AddDebugFlags
    call :AddReleaseFlags
) 
if "%2" == "reldebug" (
    call :AddDebugFlags %1
    call :AddReleaseFlags %1
) 

REM if "%1" == "release" (
REM     call :AddReleaseFlags
REM ) 
REM if "%2" == "release" (
REM )

set DefaultCompilerFlags=-FC -Zi -EHsc -nologo -Oi -WX -W4 -wd4305 -wd4324 -wd4127 -wd4700 -wd4701 -wd4505 -wd4189 -wd4201 -wd4100
set DefaultLinkerFlags= -STACK:0x100000,0x100000 -incremental:no -opt:ref 
REM user32.lib gdi32.lib ole32.lib winmm.lib Advapi32.lib

set dir="W:\rt"
pushd %dir%
IF NOT EXIST build mkdir build

pushd build
if "%1" == "cl" (
    echo Compiling with Clang
    clang++ -std=c++17 -march=native %Definitions% -I %Dependencies% -o rt.exe ../src/rt/rt.cpp
) else (
    echo Compiling with MSVC
    cl %DefaultCompilerFlags% %Definitions% %AVX2% -I%Dependencies% ../src/rt/rt.cpp /std:c++17 /link %DefaultLinkerFlags% /out:rt.exe
)

REM cl %DefaultCompilerFlags% ../src/rgb2spec.cpp /std:c++17 /link %DefaultLinkerFlags% /out:rgb2spec.exe
popd

REM clang++ -std=c++17 -march=native -O3 -l Advapi32.lib -o rt.exe ../src/rt.cpp
REM clang++ -std=c++17 -g -march=native -O3 -c ../src/rt.cpp -o rt.o

cd "src"
w:\cloc.exe --fullpath --exclude-dir=third_party,tables *
cd ..

popd

exit /b

:AddDebugFlags
set Definitions=%Definitions% -D DEBUG -D TRACK_MEMORY 
if "%1" == "cl" (
    set Definitions=%Definitions% -g
) else (
    set Definitions=%Definitions% -Od
)
exit /b

:AddReleaseFlags 
if "%1" == "cl" (
    set Definitions=%Definitions% -O3
) else (
    set Definitions=%Definitions% -O2
)
exit /b
