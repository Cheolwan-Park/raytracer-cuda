#ifndef RAYTRACER_STUDY_AABB_CUH
#define RAYTRACER_STUDY_AABB_CUH

#include "../util.cuh"
#include <utility>

class aabb {
public:
    __device__ __host__ aabb() : _min(), _max() { ; }
    __device__ __host__ explicit aabb(const vec3 &min, const vec3 &max) : _min(min), _max(max) { ; }

    __device__ bool hit(const ray &r, float tmin, float tmax) const {
        // andrew kensler's method
        for(int i=0; i<3; ++i) {
            auto invD = 1.0f / r.direction()[i];
            auto t0 = (_min[i] - r.origin()[i]) * invD;
            auto t1 = (_max[i] - r.origin()[i]) * invD;
            if (invD < 0.0f) {
                auto tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin)
                return false;
        }
        return true;
    }

    // get
    __device__ __host__ const vec3 &min() const { return _min; }
    __device__ __host__ vec3 &min() { return _min; }

    __device__ __host__ const vec3 &max() const { return _max; }
    __device__ __host__ vec3 &max() { return _max; }

    // set
    __device__ __host__ void setMin(const vec3& v) { _min = v; }

    __device__ __host__ void setMax(const vec3& v) { _max = v; }

private:
    vec3 _min;
    vec3 _max;
};

__device__ __host__ aabb surrounding_box(const aabb &a, const aabb &b) {
    vec3 min(fminf(a.min().x(), b.min().x()),
             fminf(a.min().y(), b.min().y()),
             fminf(a.min().z(), b.min().z()));
    vec3 max(fmaxf(a.max().x(), b.max().x()),
             fmaxf(a.max().y(), b.max().y()),
             fmaxf(a.max().z(), b.max().z()));
    return aabb(min, max);
}

__device__ bool box_compare(const aabb &a, const aabb &b, int axis) {
    return a.min()[axis] < b.min()[axis];
}

#endif //RAYTRACER_STUDY_AABB_CUH
