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

# DLSS SDK Configuration (MUST be before SOURCES list)
set(DLSS_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dlss")
if(NOT EXISTS "${DLSS_SDK_DIR}/include/nvsdk_ngx.h")
    message(WARNING "DLSS SDK not found at ${DLSS_SDK_DIR}. DLSS features will be disabled.")
    set(ENABLE_DLSS OFF)
else()
    message(STATUS "DLSS SDK found: ${DLSS_SDK_DIR}")
    set(ENABLE_DLSS ON)
endif()

# ONNX Runtime configuration
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime")
if(NOT EXISTS "${ONNXRUNTIME_DIR}/include" OR NOT EXISTS "${ONNXRUNTIME_DIR}/lib")
    message(WARNING "ONNX Runtime not found at ${ONNXRUNTIME_DIR}. ML features will be disabled.")
    set(ENABLE_ML_FEATURES OFF)
else()
    message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_DIR}")
    set(ENABLE_ML_FEATURES ON)
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
    src/config/Config.cpp
    src/particles/ParticleRenderer_Billboard.cpp
    src/particles/ParticleRenderer_Gaussian.cpp
    src/particles/LuminousParticleSystem.cpp
    src/ml/AdaptiveQualitySystem.cpp
    src/ml/PINNPhysicsSystem.cpp
    src/ml/SIRENVortexField.cpp
    src/benchmark/BenchmarkRunner.cpp
    src/lighting/RTLightingSystem.cpp
    src/lighting/RTXDILightingSystem.cpp
    src/lighting/RTLightingSystem_RayQuery.cpp
    src/lighting/ProbeGridSystem.cpp
    src/rendering/NanoVDBSystem.cpp
    src/utils/Logger.cpp

    src/utils/ResourceManager.cpp
    src/debug/PIXCaptureHelper.cpp
    # ImGui
    external/imgui/imgui.cpp
    external/imgui/imgui_demo.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/backends/imgui_impl_win32.cpp
    external/imgui/backends/imgui_impl_dx12.cpp
)

# Add DLSS sources if enabled
if(ENABLE_DLSS)
    list(APPEND SOURCES src/dlss/DLSSSystem.cpp)
endif()

set(HEADERS
    src/core/Application.h
    src/core/Device.h
    src/core/SwapChain.h
    src/core/FeatureDetector.h
    src/particles/ParticleSystem.h
    src/particles/ParticleRenderer.h
    src/config/Config.h
    src/particles/ParticleRenderer_Gaussian.h
    src/particles/LuminousParticleSystem.h
    src/ml/AdaptiveQualitySystem.h
    src/ml/PINNPhysicsSystem.h
    src/benchmark/BenchmarkRunner.h
    src/benchmark/BenchmarkConfig.h
    src/benchmark/BenchmarkMetrics.h
    src/lighting/RTLightingSystem.h
    src/lighting/RTLightingSystem_RayQuery.h
    src/lighting/ProbeGridSystem.h
    src/rendering/NanoVDBSystem.h
    src/utils/Logger.h

    src/utils/ResourceManager.h
    src/utils/d3dx12.h
    src/debug/PIXCaptureHelper.h
)

# Add DLSS headers if enabled
if(ENABLE_DLSS)
    list(APPEND HEADERS src/dlss/DLSSSystem.h)
endif()

# Add RTXDI SDK library
add_subdirectory(external/RTXDI-Runtime)

# Create executable
add_executable(${PROJECT_NAME} WIN32 ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/external/RTXDI-Runtime/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/nanovdb
    ${ONNXRUNTIME_DIR}/include
    "C:/Program Files/Microsoft PIX/Include"
    $<$<BOOL:${ENABLE_DLSS}>:${DLSS_SDK_DIR}/include>
)

# Compiler definitions for PIX support
target_compile_definitions(${PROJECT_NAME} PRIVATE
    USE_PIX
    UNICODE
    _UNICODE
    NOMINMAX
    $<$<BOOL:${ENABLE_ML_FEATURES}>:ENABLE_ML_FEATURES>
    $<$<BOOL:${ENABLE_DLSS}>:ENABLE_DLSS>
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    d3d12.lib
    dxgi.lib
    d3dcompiler.lib
    dxguid.lib
    Rtxdi
    $<$<BOOL:${ENABLE_ML_FEATURES}>:onnxruntime.lib>
)

# Set ONNX Runtime library directory
if(ENABLE_ML_FEATURES)
    target_link_directories(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_DIR}/lib)
endif()

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
        COMMAND dxc.exe -T ${SHADER_TYPE} -E main ${HLSL} -Fo ${OUTPUT_FILE} -I "${FILE_DIR}" -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        DEPENDS ${HLSL}
        COMMENT "Compiling ${FILE_NAME}.hlsl -> ${FILE_NAME}.dxil"
    )

    list(APPEND SHADER_OUTPUTS ${OUTPUT_FILE})
endforeach()

# Add DXR raytracing shaders (lib_6_3)
set(DXR_SHADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rtxdi/rtxdi_raygen.hlsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rtxdi/rtxdi_miss.hlsl"
)

foreach(DXR_SHADER ${DXR_SHADERS})
    get_filename_component(FILE_NAME ${DXR_SHADER} NAME_WE)
    get_filename_component(FILE_DIR ${DXR_SHADER} DIRECTORY)
    get_filename_component(DIR_NAME ${FILE_DIR} NAME)

    set(OUTPUT_FILE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/${DIR_NAME}/${FILE_NAME}.dxil")

    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/${DIR_NAME}"
        COMMAND dxc.exe -T lib_6_3 ${DXR_SHADER} -Fo ${OUTPUT_FILE} -I "${FILE_DIR}" -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        DEPENDS ${DXR_SHADER}
        COMMENT "Compiling DXR shader ${FILE_NAME}.hlsl -> ${FILE_NAME}.dxil"
    )

    list(APPEND SHADER_OUTPUTS ${OUTPUT_FILE})
endforeach()

# Compile missing shaders that don't match naming pattern
# generate_particle_aabbs (compute shader for RT AABB generation)
add_custom_command(
    OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/dxr/generate_particle_aabbs.dxil"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/dxr"
    COMMAND dxc.exe -T cs_6_5 -E main "${CMAKE_CURRENT_SOURCE_DIR}/shaders/dxr/generate_particle_aabbs.hlsl" -Fo "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/dxr/generate_particle_aabbs.dxil" -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/dxr/generate_particle_aabbs.hlsl"
    COMMENT "Compiling generate_particle_aabbs.hlsl"
)
list(APPEND SHADER_OUTPUTS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/dxr/generate_particle_aabbs.dxil")

# particle_physics (compute shader for GPU particle physics simulation)
add_custom_command(
    OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/particles/particle_physics.dxil"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/particles"
    COMMAND dxc.exe -T cs_6_5 -E main "${CMAKE_CURRENT_SOURCE_DIR}/shaders/particles/particle_physics.hlsl" -Fo "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/particles/particle_physics.dxil" -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders/particles"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/particles/particle_physics.hlsl"
    COMMENT "Compiling particle_physics.hlsl"
)
list(APPEND SHADER_OUTPUTS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/particles/particle_physics.dxil")

# particle_gaussian_raytrace (compute shader for volumetric 3D Gaussian rendering)
add_custom_command(
    OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/particles/particle_gaussian_raytrace.dxil"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/particles"
    COMMAND dxc.exe -T cs_6_5 -E main "${CMAKE_CURRENT_SOURCE_DIR}/shaders/particles/particle_gaussian_raytrace.hlsl" -Fo "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/particles/particle_gaussian_raytrace.dxil" -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders/particles"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/particles/particle_gaussian_raytrace.hlsl"
    COMMENT "Compiling particle_gaussian_raytrace.hlsl"
)
list(APPEND SHADER_OUTPUTS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/particles/particle_gaussian_raytrace.dxil")

# compute_motion_vectors (compute shader for DLSS motion vector generation)
if(ENABLE_DLSS)
    add_custom_command(
        OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/particles/compute_motion_vectors.dxil"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/particles"
        COMMAND dxc.exe -T cs_6_5 -E main "${CMAKE_CURRENT_SOURCE_DIR}/shaders/particles/compute_motion_vectors.hlsl" -Fo "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/particles/compute_motion_vectors.dxil" -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders/particles"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/particles/compute_motion_vectors.hlsl"
        COMMENT "Compiling compute_motion_vectors.hlsl"
    )
    list(APPEND SHADER_OUTPUTS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/particles/compute_motion_vectors.dxil")
endif()

# rtxdi_temporal_accumulate (compute shader for RTXDI temporal reuse)
add_custom_command(
    OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/rtxdi/rtxdi_temporal_accumulate.dxil"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/rtxdi"
    COMMAND dxc.exe -T cs_6_5 -E main "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rtxdi/rtxdi_temporal_accumulate.hlsl" -Fo "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/rtxdi/rtxdi_temporal_accumulate.dxil"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rtxdi/rtxdi_temporal_accumulate.hlsl"
    COMMENT "Compiling rtxdi_temporal_accumulate.hlsl"
)
list(APPEND SHADER_OUTPUTS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/rtxdi/rtxdi_temporal_accumulate.dxil")

# volumetric_restir shaders REMOVED (Phase 1 Cleanup)

# probe_grid update_probes (compute shader for probe lighting updates)
add_custom_command(
    OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/probe_grid/update_probes.dxil"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/probe_grid"
    COMMAND dxc.exe -T cs_6_5 -E main "${CMAKE_CURRENT_SOURCE_DIR}/shaders/probe_grid/update_probes.hlsl" -Fo "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/probe_grid/update_probes.dxil"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/probe_grid/update_probes.hlsl"
    COMMENT "Compiling update_probes.hlsl"
)
list(APPEND SHADER_OUTPUTS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/probe_grid/update_probes.dxil")

# froxel shaders REMOVED (Phase 1 Cleanup)

# shadows depth_prepass (compute shader for screen-space shadow depth pre-pass - Phase 1)
add_custom_command(
    OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/shadows/depth_prepass.dxil"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/shadows"
    COMMAND dxc.exe -T cs_6_5 -E main "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shadows/depth_prepass.hlsl" -Fo "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/shadows/depth_prepass.dxil"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shadows/depth_prepass.hlsl"
    COMMENT "Compiling depth_prepass.hlsl (Phase 1: Screen-Space Shadows)"
)
list(APPEND SHADER_OUTPUTS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/shadows/depth_prepass.dxil")

# shadows depth_buffer_clear (compute shader for clearing shadow depth buffer - Phase 2)
add_custom_command(
    OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/shadows/depth_buffer_clear.dxil"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/shadows"
    COMMAND dxc.exe -T cs_6_5 -E main "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shadows/depth_buffer_clear.hlsl" -Fo "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/shadows/depth_buffer_clear.dxil"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shadows/depth_buffer_clear.hlsl"
    COMMENT "Compiling depth_buffer_clear.hlsl (Phase 2: Screen-Space Shadows)"
)
list(APPEND SHADER_OUTPUTS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/shadows/depth_buffer_clear.dxil")

# blit_hdr_to_sdr (vertex + pixel shader for HDR->SDR conversion)
add_custom_command(
    OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/util/blit_hdr_to_sdr_vs.dxil"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/util"
    COMMAND dxc.exe -T vs_6_5 -E VSMain "${CMAKE_CURRENT_SOURCE_DIR}/shaders/util/blit_hdr_to_sdr.hlsl" -Fo "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/util/blit_hdr_to_sdr_vs.dxil"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/util/blit_hdr_to_sdr.hlsl"
    COMMENT "Compiling blit_hdr_to_sdr_vs.hlsl"
)
list(APPEND SHADER_OUTPUTS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/util/blit_hdr_to_sdr_vs.dxil")

add_custom_command(
    OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/util/blit_hdr_to_sdr_ps.dxil"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/util"
    COMMAND dxc.exe -T ps_6_5 -E PSMain "${CMAKE_CURRENT_SOURCE_DIR}/shaders/util/blit_hdr_to_sdr.hlsl" -Fo "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/util/blit_hdr_to_sdr_ps.dxil"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/util/blit_hdr_to_sdr.hlsl"
    COMMENT "Compiling blit_hdr_to_sdr_ps.hlsl"
)
list(APPEND SHADER_OUTPUTS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/util/blit_hdr_to_sdr_ps.dxil")

# NanoVDB volumetric ray marching (compute shader for sparse volumetric fog/gas rendering)
add_custom_command(
    OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/volumetric/nanovdb_raymarch.dxil"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/volumetric"
    COMMAND dxc.exe -T cs_6_5 -E main "${CMAKE_CURRENT_SOURCE_DIR}/shaders/volumetric/nanovdb_raymarch.hlsl" -Fo "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/volumetric/nanovdb_raymarch.dxil" -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders" -I "${CMAKE_CURRENT_SOURCE_DIR}/external/nanovdb"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/volumetric/nanovdb_raymarch.hlsl" "${CMAKE_CURRENT_SOURCE_DIR}/external/nanovdb/PNanoVDB.h"
    COMMENT "Compiling nanovdb_raymarch.hlsl (NanoVDB volumetric fog/gas rendering)"
)
list(APPEND SHADER_OUTPUTS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/volumetric/nanovdb_raymarch.dxil")

add_custom_target(CompileShaders ALL DEPENDS ${SHADER_OUTPUTS})
add_dependencies(${PROJECT_NAME} CompileShaders)

# Copy ONNX Runtime DLLs to output
if(ENABLE_ML_FEATURES)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/"
        COMMENT "Copying ONNX Runtime DLLs to output directory"
    )
endif()

# Copy config files to build directory for easy access
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/configs
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/configs
    COMMENT "Copying config files to build directory"
)


# Link DLSS SDK library
if(ENABLE_DLSS)
    target_link_libraries(${PROJECT_NAME}
        ${DLSS_SDK_DIR}/lib/Windows_x86_64/x64/nvsdk_ngx_d$<$<CONFIG:Debug>:_dbg>.lib
    )
endif()

# DLSS DLL Handling: Bundle REL DLLs to ensure compatibility
#
# FIX: Bundling 'rel' (Release) DLLs is required because:
# 1. The driver's DriverStore is missing 'nvngx_dlss.dll' (Super Resolution).
# 2. The 'dev' DLLs fail signature validation (0xBAD00014) with the production driver loader.
# 3. Release DLLs are properly signed and accepted by the driver's loader.
if(ENABLE_DLSS)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLSS_SDK_DIR}/lib/Windows_x86_64/rel/nvngx_dlss.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLSS_SDK_DIR}/lib/Windows_x86_64/rel/nvngx_dlssd.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLSS_SDK_DIR}/lib/Windows_x86_64/rel/nvngx_dlssg.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/"
        COMMENT "Copying DLSS DLLs (rel) to output directory"
    )
endif()
