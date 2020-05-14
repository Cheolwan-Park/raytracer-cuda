#ifndef RAYTRACER_STUDY_RAY_CUH
#define RAYTRACER_STUDY_RAY_CUH

#include "util.cuh"

class ray {
public:
    __device__ ray() : A(), B() {}
    __device__ explicit ray(const vec3 &a, const vec3 &b) : A(a), B(b) { ; }
    __device__ vec3 origin() const       { return A; }
    __device__ vec3 direction() const    { return B; }
    __device__ vec3 at(float t) const { return A + t*B; }

    vec3 A;
    vec3 B;
};

#endif //RAYTRACER_STUDY_RAY_CUH
