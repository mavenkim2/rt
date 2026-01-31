# 1. Search for the include directory containing optix.h
find_path(OPTIX_INCLUDE_DIR
    NAMES optix.h
    PATHS 
        "${OPTIX_ROOT_DIR}"
        "$ENV{OPTIX_ROOT_DIR}"
    PATH_SUFFIXES include
    DOC "Path to OptiX include directory"
)

# 2. Extract Version from optix.h
if(OPTIX_INCLUDE_DIR AND EXISTS "${OPTIX_INCLUDE_DIR}/optix.h")
    file(READ "${OPTIX_INCLUDE_DIR}/optix.h" _OPTIX_HEADER_CONTENTS)
    
    # OptiX defines version as: #define OPTIX_VERSION 70500
    string(REGEX MATCH "#define OPTIX_VERSION ([0-9]+)" _MATCH "${_OPTIX_HEADER_CONTENTS}")
    
    if(_MATCH)
        set(OPTIX_VERSION "${CMAKE_MATCH_1}")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX
    REQUIRED_VARS OPTIX_INCLUDE_DIR
    VERSION_VAR OPTIX_VERSION
)

mark_as_advanced(
    OPTIX_INCLUDE_DIR
    OPTIX_VERSION
)
