cmake_minimum_required(VERSION 3.17)
project(raytracer_study CUDA)

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 14)

add_executable(
        raytracer_study
        src/main.cu
        src/hittables/hittable.cuh src/hittables/hittable_list.cuh src/hittables/sphere.cuh src/material.cuh src/ray.cuh src/util.cuh
        src/vec3.cuh src/camera.cuh src/hittables/aabb.cuh src/hittables/bvh_node.cuh src/texture.cuh
)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

set_target_properties(
        raytracer_study
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(
        raytracer_study
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)