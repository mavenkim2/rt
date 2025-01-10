@echo off

pushd %~dp0
set SSE2=-D __SSE__ -D __SSE2__
set SSE42=%SSE2% -D __SSE3__ -D __SSSE3__ -D __SSE4_1__ -D __SSE4_2__
set AVX=%SSE42% -arch:AVX
set AVX2=%SSE42% -arch:AVX2

set Definitions=-D NOMINMAX -D PTEX_STATIC

if not exist .\src\gen\rgbspectrum_srgb.cpp (
    IF NOT EXIST build mkdir build
    pushd build 
    clang++ -std=c++17 -march=native -D NOMINMAX -O3 -o rgb2spec.exe ../src/rt/rgb2spec.cpp
    IF NOT EXIST ..\src\gen (
        pushd ..\src
        mkdir gen
        popd
    )
    
    echo Generating rgbspectrum_srgb tables
    .\rgb2spec.exe ../src/gen/rgbspectrum_srgb.cpp
    popd
)

if not exist .\src\gen\rgbspectrum_srgb.cpp (
    echo Could not generate srgb table
    exit /b
)

if not exist .\build\src\third_party\zlib\Release\zlibstatic.lib (
    REM TODO this probably only works with visual studio...
    cmake -B build -T ClangCL -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded . && cmake --build build --config Release
)

set Dependencies=-I ..\src\third_party\openvdb\nanovdb -I ..\src\third_party\ptex\src\ptex -I ..\src\third_party\zlib ^
-I .\src\third_party\zlib
set LibraryPathPtex=.\src\third_party\ptex\src\ptex\Release
set LibraryPathZlib=.\src\third_party\zlib\Release
set LibraryNamePtex=Ptex.lib 
set LibraryNameZlib=zlibstatic.lib

if "%1" == "release" (
    call :AddReleaseFlags
) else (
    if "%2" == "release" (
        call :AddReleaseFlags %1
    ) else (
        if "%1" == "reldebug" (
            call :AddRelDebugFlags
        ) else (
            if "%2" == "reldebug" (
                call :AddRelDebugFlags %1
            ) else (
                call :AddDebugFlags %1
            )
        )
    )
) 

set DefaultCompilerFlags=-FC -Zi -EHsc -nologo -Oi -WX -W4 -wd4305 -wd4324 -wd4127 -wd4700 -wd4701 -wd4505 -wd4189 -wd4201 -wd4100
set DefaultLinkerFlags= -STACK:0x100000,0x100000 -incremental:no -opt:ref 
REM user32.lib gdi32.lib ole32.lib winmm.lib Advapi32.lib

REM set dir="W:\rt"
REM pushd %dir%
IF NOT EXIST build mkdir build

pushd build

if not exist .\rgbspectrum_srgb.obj ( 
    REM todo msvc?
    clang++ -fms-runtime-lib=dynamic -fms-compatibility -std=c++17 -march=native -ffp-contract=off -O3 -c ..\src\gen\rgbspectrum_srgb.cpp -o rgbspectrum_srgb.obj
)

if not exist .\rgbspectrum_srgb.obj (
    echo Could not compile sRGB to spectrum tables
    exit /b
)

if "%1" == "cl" (
    echo Compiling with Clang
    clang++ -std=c++17 -ferror-limit=5 -march=native -ffp-contract=off %Definitions% %Dependencies% -L %LibraryPathZlib% -L %LibraryPathPtex% -l %LibraryNameZlib% -l %LibraryNamePtex% -o rt.exe ../src/rt/rt.cpp rgbspectrum_srgb.obj
) else (
    echo Compiling with MSVC
    cl %DefaultCompilerFlags% %Definitions% %AVX2% %Dependencies% ../src/rt/rt.cpp /std:c++17 /link %DefaultLinkerFlags% rgbspectrum_srgb.obj /LIBPATH:%LibraryPathZlib% /LIBPATH:%LibraryPathPtex% %LibraryNameZlib% %LibraryNamePtex% /out:rt.exe
)

REM cl %DefaultCompilerFlags% ../src/rgb2spec.cpp /std:c++17 /link %DefaultLinkerFlags% /out:rgb2spec.exe
popd

if exist w:\cloc.exe (
cd "src"
w:\cloc.exe --fullpath --exclude-dir=third_party,tables,gen *
cd ..
)

popd
exit /b

:AddDebugFlags
echo Compiling Debug
set Definitions=%Definitions% -D DEBUG -D TRACK_MEMORY 
if "%1" == "cl" (
    set Definitions=%Definitions% -g
) else (
    set Definitions=%Definitions% -Od
)
exit /b

:AddReleaseFlags 
echo Compiling Release
if "%1" == "cl" (
    set Definitions=%Definitions% -O3
) else (
    set Definitions=%Definitions% -O2
)
exit /b

:AddRelDebugFlags 
echo Compiling RelDebug (currently only works for clang)
set Definitions=%Definitions% -D DEBUG -D TRACK_MEMORY
if "%1" == "cl" (
    set Definitions=%Definitions% -O3 -g
) else (
    set Definitions=%Definitions% -O2
)
exit /b
