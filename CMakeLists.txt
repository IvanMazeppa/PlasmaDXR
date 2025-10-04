cmake_minimum_required(VERSION 3.20)
project(PlasmaDX-Clean)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Platform check
if(NOT WIN32)
    message(FATAL_ERROR "PlasmaDX-Clean requires Windows")
endif()

# Find Windows SDK
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ARCH "x64")
else()
    set(ARCH "x86")
endif()

# Sources
set(SOURCES
    src/main.cpp
    src/core/Application.cpp
    src/core/Device.cpp
    src/core/SwapChain.cpp
    src/core/FeatureDetector.cpp
    src/particles/ParticleSystem.cpp
    src/particles/ParticleRenderer.cpp
    src/particles/ParticlePhysics.cpp
    src/lighting/RTLightingSystem.cpp
    src/lighting/AccelerationStructure.cpp
    src/utils/Logger.cpp
    src/utils/ShaderManager.cpp
    src/utils/ResourceManager.cpp
)

set(HEADERS
    src/core/Application.h
    src/core/Device.h
    src/core/SwapChain.h
    src/core/FeatureDetector.h
    src/particles/ParticleSystem.h
    src/particles/ParticleRenderer.h
    src/particles/ParticlePhysics.h
    src/lighting/RTLightingSystem.h
    src/lighting/AccelerationStructure.h
    src/utils/Logger.h
    src/utils/ShaderManager.h
    src/utils/ResourceManager.h
    src/utils/d3dx12.h
)

# Create executable
add_executable(${PROJECT_NAME} WIN32 ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    d3d12.lib
    dxgi.lib
    d3dcompiler.lib
    dxguid.lib
)

# Set startup project for VS
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

# Copy Agility SDK to output
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${PROJECT_NAME}>/D3D12"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/external/D3D12/D3D12Core.dll"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/D3D12/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/external/D3D12/d3d12SDKLayers.dll"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/D3D12/"
)

# Shader compilation
file(GLOB_RECURSE HLSL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.hlsl")

foreach(HLSL ${HLSL_FILES})
    get_filename_component(FILE_NAME ${HLSL} NAME_WE)
    get_filename_component(FILE_DIR ${HLSL} DIRECTORY)
    get_filename_component(DIR_NAME ${FILE_DIR} NAME)

    set(OUTPUT_FILE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/${DIR_NAME}/${FILE_NAME}.dxil")

    # Determine shader type from filename
    if(${FILE_NAME} MATCHES ".*_vs$")
        set(SHADER_TYPE "vs_6_5")
    elseif(${FILE_NAME} MATCHES ".*_ps$")
        set(SHADER_TYPE "ps_6_5")
    elseif(${FILE_NAME} MATCHES ".*_cs$" OR ${FILE_NAME} MATCHES ".*_compute$")
        set(SHADER_TYPE "cs_6_5")
    elseif(${FILE_NAME} MATCHES ".*_ms$" OR ${FILE_NAME} MATCHES ".*_mesh$")
        set(SHADER_TYPE "ms_6_5")
    elseif(${FILE_NAME} MATCHES ".*_as$")
        set(SHADER_TYPE "as_6_5")
    elseif(${FILE_NAME} MATCHES ".*_lib$")
        set(SHADER_TYPE "lib_6_3")
    else()
        continue()
    endif()

    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/${DIR_NAME}"
        COMMAND dxc.exe -T ${SHADER_TYPE} -E main ${HLSL} -Fo ${OUTPUT_FILE}
        DEPENDS ${HLSL}
        COMMENT "Compiling ${FILE_NAME}.hlsl -> ${FILE_NAME}.dxil"
    )

    list(APPEND SHADER_OUTPUTS ${OUTPUT_FILE})
endforeach()

add_custom_target(CompileShaders ALL DEPENDS ${SHADER_OUTPUTS})
add_dependencies(${PROJECT_NAME} CompileShaders)