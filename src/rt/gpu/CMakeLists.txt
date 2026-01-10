# step 1. compile to ptx
#       a. need an executable, with all flags and options set
#       b. need the source files
#       c. 
# step 2. somehow load the compiled stuff to a cu module

if (WITH_CUDA)
    if (NOT CMAKE_CUDA_COMPILER)
        message (FATAL_ERROR "CUDA compiler not found")
    else()
        set (CUDA_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/test.cu")
        set (CUDA_FILE "kernel.cubin")
        set(CUDA_OPTIONS 
            -arch=native
            -cubin
            -o ${CMAKE_CURRENT_BINARY_DIR}/${CUDA_FILE}
            ${CUDA_KERNEL_SRC}
        )
        add_custom_command(
            OUTPUT ${CUDA_FILE}
            COMMAND ${CMAKE_CUDA_COMPILER} ${CUDA_OPTIONS}
            DEPENDS ${CUDA_KERNEL_SRC})
        add_custom_target(cuda_kernels ALL DEPENDS ${CUDA_FILE})
    endif()
endif()
