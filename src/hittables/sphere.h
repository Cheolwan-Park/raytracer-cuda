#ifndef RAYTRACER_STUDY_SPHERE_H
#define RAYTRACER_STUDY_SPHERE_H

#include "hittable.h"
#include "../util.h"

class sphere : public hittable {
public:
    CUDA_CALLABLE_MEMBER sphere(): _pos(), _radius(0.0f), _mat(nullptr) { ; }
    CUDA_CALLABLE_MEMBER explicit sphere(const vec3 &pos, float radius, material *mat)
                                        : _pos(pos), _radius(radius), _mat(mat) { ; }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    CUDA_CALLABLE_MEMBER void freeMat() { delete _mat; }

    // get
    CUDA_CALLABLE_MEMBER const vec3 &pos() const { return _pos; }
    CUDA_CALLABLE_MEMBER vec3 &pos() { return _pos; }

    CUDA_CALLABLE_MEMBER float getRadius() const { return _radius; }

    CUDA_CALLABLE_MEMBER const material *getMaterial() { return _mat; }

    // set
    CUDA_CALLABLE_MEMBER void setPos(const vec3 &v) { _pos = v; }

    CUDA_CALLABLE_MEMBER void setRadius(float v) { _radius = v; }

    CUDA_CALLABLE_MEMBER void setMaterial(material *mat) { _mat = mat; }
private:
    vec3 _pos;
    float _radius;
    material *_mat;
};

__device__ bool sphere::hit(const ray &r, float t_min, float t_max, hit_record &rec) const {
    vec3 oc = r.origin() - _pos;
    auto a = dot(r.direction(), r.direction());
    auto b = dot(oc, r.direction());
    auto c = dot(oc, oc) - _radius*_radius;
    auto D = b*b - a*c;
    if(D > 0) {
        float temp = (-b - sqrt(D))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - _pos) / _radius;
            rec.mat_ptr = _mat;
            return true;
        }
        temp = (-b + sqrt(D)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - _pos) / _radius;
            rec.mat_ptr = _mat;
            return true;
        }
    }
    return false;
}

#endif //RAYTRACER_STUDY_SPHERE_H
