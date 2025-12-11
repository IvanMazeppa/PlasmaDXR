
file(GLOB sources "*.cpp" "*.h")

set(project MinimalSample)
set(folder "RTXDI SDK")

include(CMakeDependentOption)

add_executable(${project} WIN32 ${sources})

target_link_libraries(${project} donut_core donut_engine donut_app donut_render Rtxdi cxxopts)
add_dependencies(${project} MinimalSampleShaders)
set_target_properties(${project} PROPERTIES FOLDER ${folder})
