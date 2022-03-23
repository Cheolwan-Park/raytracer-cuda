#ifndef RAYTRACER_STUDY_SPHERE_CUH
#define RAYTRACER_STUDY_SPHERE_CUH

#include "hittable.cuh"
#include "../util.cuh"

class sphere : public hittable {
public:
    __device__ sphere(): _pos(), _radius(0.0f), _mat(nullptr) { ; }
    __device__ explicit sphere(const vec3 &pos, float radius, base_material *mat)
                                        : _pos(pos), _radius(radius), _mat(mat) { ; }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ __host__ void bounding_box(aabb &output) const override;

    __device__ void freeMat() { delete _mat; }

    // get
    __device__ const vec3 &pos() const { return _pos; }
    __device__ vec3 &pos() { return _pos; }

    __device__ float getRadius() const { return _radius; }

    __device__ const base_material *getMaterial() { return _mat; }

    // set
    __device__ void setPos(const vec3 &v) { _pos = v; }

    __device__ void setRadius(float v) { _radius = v; }

    __device__ void setMaterial(base_material *mat) { _mat = mat; }
private:
    vec3 _pos;
    float _radius;
    base_material *_mat;
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

__device__ __host__ void sphere::bounding_box(aabb &output) const {
    vec3 r(_radius, _radius, _radius);
    output = aabb(_pos - r, _pos + r);
}

#endif //RAYTRACER_STUDY_SPHERE_CUH
