#ifndef RAYTRACER_STUDY_HITTABLE_CUH
#define RAYTRACER_STUDY_HITTABLE_CUH

#include "../util.cuh"
#include "../ray.cuh"
#include "aabb.cuh"
#include <iostream>

class base_material;

struct hit_record {
    vec3 p;
    vec3 normal;
    float t;
    float u;
    float v;
    base_material *mat_ptr;
};

class hittable {
public:
    __device__ __host__ virtual ~hittable() { ; }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;

    __device__ __host__ virtual void bounding_box(aabb &output) const = 0;
};

__device__ __host__ bool hittable_box_compare(const hittable &a, const hittable &b, int axis) {
    aabb box_a, box_b;
    a.bounding_box(box_a);
    b.bounding_box(box_b);
    return box_a.min()[axis] < box_b.min()[axis];
}

#endif //RAYTRACER_STUDY_HITTABLE_CUH
