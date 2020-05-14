#ifndef RAYTRACER_STUDY_HITTABLE_LIST_CUH
#define RAYTRACER_STUDY_HITTABLE_LIST_CUH

#include "hittable.cuh"
#include "../util.cuh"
#include <thrust/device_vector.h>
#include <thrust/memory.h>

class hittable_list : public hittable {
public:
    __device__ hittable_list() : _list(nullptr), _size(0) { ; }
    __device__ explicit hittable_list(hittable **list, unsigned int size) : _list(list), _size(size) { ; }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ __host__ void bounding_box(aabb &output) const override;

    // sort [s, e-1]
    __device__ void qsort(int s, int e, int axis, int *stack);

    // get
    __device__ const hittable *get(unsigned int idx) const { return _list[idx]; }
    __device__ hittable *get(unsigned int idx) { return _list[idx]; }

    __device__ unsigned int size() const { return _size; }

    // set
    __device__ void set(hittable **list, unsigned int size) { _list = list; _size = size; }

private:
    __device__ int qsort_partition(int s, int e, int axis);

private:
    hittable **_list;
    unsigned int _size;
};

__device__ bool hittable_list::hit(const ray &r, float t_min, float t_max, hit_record &rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < _size; i++) {
        if (_list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

__device__ __host__ void hittable_list::bounding_box(aabb &output) const {aabb tmp;
    bool first = true;

    for(int i=0; i<_size; ++i) {
        _list[i]->bounding_box(tmp);
        output = first ? tmp : surrounding_box(output, tmp);
        first = false;
    }
}

__device__ void hittable_list::qsort(int s, int e, int axis, int *stack) {
    int top = -1;
    stack[++top] = s;
    stack[++top] = e-1;

    while(top > 0) {
        e = stack[top--];
        s = stack[top--];

        int pivot = qsort_partition(s, e, axis);
        if(pivot-1 > s) {   // left side exist
            stack[++top] = s;
            stack[++top] = pivot-1;
        }
        if(pivot+1 < e) {   // right side exist
            stack[++top] = pivot+1;
            stack[++top] = e;
        }
    }
}

__device__ int hittable_list::qsort_partition(int s, int e, int axis) {
    const hittable *pivot = _list[e];
    int i = (s-1);
    for(int j=s; j<e; ++j) {
        if(hittable_box_compare(*_list[j], *pivot, axis)) {
            ++i;
            gpu_swap(_list[i], _list[j]);
        }
    }
    gpu_swap(_list[i+1], _list[e]);
    return i+1;
}

#endif //RAYTRACER_STUDY_HITTABLE_LIST_CUH
