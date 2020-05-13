#ifndef RAYTRACER_STUDY_RAY_H
#define RAYTRACER_STUDY_RAY_H

#include "util.h"

class ray
{
public:
    CUDA_CALLABLE_MEMBER ray() : A(), B() {}
    CUDA_CALLABLE_MEMBER explicit ray(const vec3 &a, const vec3 &b) : A(a), B(b) { ; }
    CUDA_CALLABLE_MEMBER vec3 origin() const       { return A; }
    CUDA_CALLABLE_MEMBER vec3 direction() const    { return B; }
    CUDA_CALLABLE_MEMBER vec3 at(float t) const { return A + t*B; }

    vec3 A;
    vec3 B;
};

#endif //RAYTRACER_STUDY_RAY_H
