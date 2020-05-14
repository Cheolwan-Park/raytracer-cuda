#ifndef RAYTRACER_STUDY_BVH_NODE_CUH
#define RAYTRACER_STUDY_BVH_NODE_CUH

#include "hittable.cuh"
#include "hittable_list.cuh"
#include "../util.cuh"
#include "sphere.cuh"

enum class bvh_node_type : int {
    NONE = -1,
    LEAF = 0,
    PARENT = 1
};

struct bvh_node {
    bvh_node_type type;
    hittable *obj;
    aabb box;
};

__global__ void construct_bvh_node(hittable_list **l, bvh_node *bvh_tree) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        hittable_list &list = **l;
        if(0 == list.size()) return;
        curandState rand_state;
        curand_init(1984, 0, 0, &rand_state);

        // build tree
        int *qsort_stack = new int[list.size()];
        int *stack = new int[list.size()*3];
        int top = -1;
        stack[++top] = 1;                   // tree idx
        stack[++top] = 0;                   // start
        stack[++top] = int(list.size()) -1; // end
        while(top > 0) {
            int e = stack[top--], s = stack[top--], idx = stack[top--];
            int axis = int(gpu_random(&rand_state, 0, 2.9999f));
            size_t span = e - s;
            if(span == 1) {
                bvh_tree[idx].type = bvh_node_type::LEAF;
                bvh_tree[idx].obj = list.get(s);
                bvh_tree[idx].obj->bounding_box(bvh_tree[idx].box);
            } else if(span == 2) {
                bvh_tree[idx].type = bvh_node_type::PARENT;
                if(hittable_box_compare(*list.get(s), *list.get(s+1), axis)) {
                    stack[++top] = idx*2; stack[++top] = s; stack[++top] = s+1;     // left
                    stack[++top] = idx*2+1; stack[++top] = s+1; stack[++top] = s+2; // right
                } else {
                    stack[++top] = idx*2; stack[++top] = s+1; stack[++top] = s+2;   // left
                    stack[++top] = idx*2+1; stack[++top] = s; stack[++top] = s+1;   // right
                }
            } else {
                list.qsort(s, e, axis, qsort_stack);

                auto mid = s + span/2;
                bvh_tree[idx].type = bvh_node_type::PARENT;
                stack[++top] = idx*2; stack[++top] = s; stack[++top] = mid;     // left
                stack[++top] = idx*2+1; stack[++top] = mid; stack[++top] = e;   // right
            }
        }

        // print
//        stack[++top] = 1;   // idx
//        stack[++top] = 0;   // state
//        while(top >= 0) {
//            int state = stack[top--], idx = stack[top--];
//            if(0 == state) {    // first encounter
//                if(bvh_tree[idx].type == bvh_node_type::LEAF) {
//                    auto *obj = (sphere*)bvh_tree[idx].obj;
//                    printf("%d(%f, %f, %f) ", idx, obj->pos().x(), obj->pos().y(), obj->pos().z());
//                } else if(bvh_tree[idx].type == bvh_node_type::PARENT) {
//                    stack[++top] = idx; stack[++top] = 1;
//                    stack[++top] = idx*2+1; stack[++top] = 0; // right
//                    stack[++top] = idx*2; stack[++top] = 0; // left
//                    printf("%d{ ", idx);
//                }
//            } else {
//                printf("} ");
//            }
//        }
//        printf("\n");

        stack[++top] = 1;   // idx;
        stack[++top] = 0;   // state
        while(top > 0) {
            int idx = stack[top--], state = stack[top--];
            if(0 == state) {
                if (bvh_tree[idx].type == bvh_node_type::LEAF) {
                    bvh_tree[idx].obj->bounding_box(bvh_tree[idx].box);
                } else if(bvh_tree[idx].type == bvh_node_type::PARENT) {
                    stack[++top] = idx; stack[++top] = 1;
                    stack[++top] = idx*2+1; stack[++top] = 0;
                    stack[++top] = idx*2; stack[++top] = 0;
                }
            } else {
                bvh_tree[idx].box = surrounding_box(bvh_tree[idx*2].box, bvh_tree[idx*2+1].box);
            }
        }

        delete[] stack;
        delete[] qsort_stack;
    }
}

__device__ bool hit_bvh_tree(const ray& r, float t_min, float t_max, hit_record& rec, bvh_node *bvh_tree, int *stack) {
    int top = -1;
    bool hit_anything = false;
    float closest_so_far = t_max;
    hit_record tmp_rec;

    stack[++top] = 1;

    while(top >= 0) {
        int now = stack[top--];
        if(bvh_tree[now].type == bvh_node_type::LEAF
        && bvh_tree[now].box.hit(r, t_min, closest_so_far)
        && bvh_tree[now].obj->hit(r, t_min, closest_so_far, tmp_rec)) {
            hit_anything = true;
            closest_so_far = tmp_rec.t;
            rec = tmp_rec;
        } else if(bvh_tree[now].type == bvh_node_type::PARENT) {
            if(bvh_tree[now].box.hit(r, t_min, closest_so_far)) {
                stack[++top] = now*2;       // left
                stack[++top] = now*2+1;     // right
            }
        }
    }
    return hit_anything;
}

#endif //RAYTRACER_STUDY_BVH_NODE_CUH
