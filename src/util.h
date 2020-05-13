#ifndef RAYTRACER_STUDY_UTIL_H
#define RAYTRACER_STUDY_UTIL_H

#include <fstream>
#include <string>
#include <curand_kernel.h>
#include <limits>
#include "vec3.h"

constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi = 3.1415926535897932385f;

__device__ __host__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}

__device__ inline float gpu_random(curandState *rand_state, float min, float max) {
    return curand_uniform(rand_state)*(max-min) + min;
}
__device__ inline vec3 random_vec3(curandState *rand_state, float min, float max) {
    return vec3(gpu_random(rand_state, min, max), gpu_random(rand_state, min, max), gpu_random(rand_state, min, max));
}
__device__ inline vec3 random01_vec3(curandState *rand_state) {
    return vec3(curand_uniform(rand_state),curand_uniform(rand_state),curand_uniform(rand_state));
}
__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = random_vec3(local_rand_state, -1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}
__device__ vec3 random_unit_vector(curandState *rand_state) {
    auto a = gpu_random(rand_state, 0, 2*pi);
    auto z = gpu_random(rand_state, -1, 1);
    auto r = sqrt(1 - z*z);
    return vec3(r*cos(a), r*sin(a), z);
}
__device__ vec3 random_in_unit_disk(curandState *rand_state) {
//    while (true) {
//        auto p = vec3(gpu_random(rand_state, -1, 1), gpu_random(rand_state, -1, 1), 0);
//        if (p.squared_length() >= 1) continue;
//        return p;
//    }
    auto theta = gpu_random(rand_state, 0, 2*pi);
    vec3 p(cos(theta), sin(theta), 0);
    return p * curand_uniform(rand_state);
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f*dot(v,n)*n;
}

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

bool saveAsPPM(const std::string &filename, const vec3 *fb, const int nx, const int ny) {
    using namespace std;

    ofstream f(filename);
    if(!f) {
        cerr << "cannot open " << filename << endl;
        return false;
    }
    f << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            float r = fb[pixel_index].x();
            float g = fb[pixel_index].y();
            float b = fb[pixel_index].z();
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
            f << ir << " " << ig << " " << ib << " ";
        }
    }
    return true;
}

#endif //RAYTRACER_STUDY_UTIL_H
