#ifndef RAYTRACER_STUDY_HITTABLE_H
#define RAYTRACER_STUDY_HITTABLE_H

#include "../util.h"
#include "../ray.h"

class material;

struct hit_record {
    vec3 p;
    vec3 normal;
    float t;
//    bool front_face;
    material *mat_ptr;

//    CUDA_CALLABLE_MEMBER inline void set_face_normal(const ray& r, const vec3& outward_normal) {
//        front_face = dot(r.direction(), outward_normal) < 0;
//        normal = front_face ? outward_normal :-outward_normal;
//    }
};

class hittable {
public:
    CUDA_CALLABLE_MEMBER virtual ~hittable() { ; }
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif //RAYTRACER_STUDY_HITTABLE_H
