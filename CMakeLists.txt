cmake_minimum_required(VERSION 3.16)
project(raytracer_study CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(
        raytracer_study
        src/main.cu
        src/hittables/hittable.h src/hittables/hittable_list.h src/hittables/sphere.h src/material.h src/ray.h src/util.h
        src/vec3.h src/camera.h)


if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

set_target_properties(
        raytracer_study
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)