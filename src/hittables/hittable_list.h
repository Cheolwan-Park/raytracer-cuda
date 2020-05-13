#ifndef RAYTRACER_STUDY_HITTABLE_LIST_H
#define RAYTRACER_STUDY_HITTABLE_LIST_H

#include "hittable.h"
#include "../util.h"

class hittable_list : public hittable {
public:
    CUDA_CALLABLE_MEMBER hittable_list() : _list(nullptr), _size(0) { ; }
    CUDA_CALLABLE_MEMBER explicit hittable_list(hittable **list, unsigned int size) : _list(list), _size(size) { ; }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    // get
    CUDA_CALLABLE_MEMBER const hittable *get(unsigned int idx) const { return _list[idx]; }
    CUDA_CALLABLE_MEMBER hittable *get(unsigned int idx) { return _list[idx]; }

    CUDA_CALLABLE_MEMBER unsigned int size() const { return _size; }

    // set
    CUDA_CALLABLE_MEMBER void set(hittable **list, unsigned int size) { _list = list; _size = size; }

private:
    hittable **_list;
    unsigned int _size;
};

__device__ bool hittable_list::hit(const ray &r, float t_min, float t_max, hit_record &rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < _size; i++) {
        if (_list[i] && _list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

#endif //RAYTRACER_STUDY_HITTABLE_LIST_H
