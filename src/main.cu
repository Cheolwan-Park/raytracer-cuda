#include <iostream>
#include <ctime>
#include "vec3.cuh"
#include "ray.cuh"
#include "material.cuh"
#include "camera.cuh"
#include "hittables/hittable_list.cuh"
#include "hittables/bvh_node.cuh"
#include "hittables/sphere.cuh"

using namespace std;

constexpr unsigned int nx = 1920;
constexpr unsigned int ny = 1080;
constexpr unsigned int ns = 64;
constexpr unsigned int nb = 50;
constexpr unsigned int tx = 16;
constexpr unsigned int ty = 16;

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
void check_cuda(cudaError_t result, const char *func, const char *file, int line);
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

__device__ vec3 color(const ray& r, bvh_node *world, int *stack, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation(1, 1, 1);
    for(int b=0; b < nb; ++b) {
        hit_record rec;
        if(hit_bvh_tree(cur_ray, 0.001f, infinity, rec, world, stack)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else return vec3(0, 0, 0);
        } else {
            vec3 unit_direction = unit_vector(r.direction());
            auto t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 background_color = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * background_color;
        }
    }
    return vec3(0, 0, 0);
}

__global__ void render(vec3 *fb, camera **cam, bvh_node *world, curandState *rand_state, int *whole_stack) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) return;

    auto pixel_index = j*nx + i;
    curandState local_rand_state = rand_state[pixel_index];

    int *stack = whole_stack + pixel_index * bvh_stack_max_depth;

    vec3 col(0, 0, 0);
    for(int s=0; s<ns; ++s) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(nx);
        float v = float(j + curand_uniform(&local_rand_state)) / float(ny);
        ray r = (*cam)->getRay(u, v, rand_state);
        col += color(r, world, stack, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col = sqrt(col);
    fb[pixel_index] = col;
}

constexpr unsigned int world_size = 1 + 22*22 + 3;

__global__ void random_scene(hittable **gpu_list) {
    curandState rand_state;

    base_material *mat = new lambertian(new solid_texture(vec3(0.5, 0.5, 0.5)));
    gpu_list[0] = new sphere(vec3(0, -1000, 0), 1000, mat);

    int idx=1;
    for(int x=-11; x<11; ++x) {
        for(int y=-11; y<11; ++y) {
            auto choose_mat = gpu_random(&rand_state, 0, 1);
            vec3 center((float)x + gpu_random(&rand_state, 0, 1)*0.9f, 0.2, (float)y + gpu_random(&rand_state, 0, 1)*0.9f);

            if(choose_mat < 0.8) {
                auto albedo = random01_vec3(&rand_state) * random01_vec3(&rand_state);
                mat = new lambertian(new solid_texture(albedo));
            } else if(choose_mat < 0.95) {
                auto albedo = random_vec3(&rand_state, 0.5, 1);
                auto fuzz = gpu_random(&rand_state, 0, 0.5);
                mat = new metal(albedo, fuzz);
            } else {
                auto albedo = random_vec3(&rand_state, 0.5, 1);
                mat = new dielectric(albedo, 1.5);
            }
            gpu_list[idx++] = new sphere(center, 0.2, mat);
        }
    }

    auto mat1 = new dielectric(vec3(1, 1, 1), 1.5);
    gpu_list[idx++] = new sphere(vec3(0, 1, 0), 1, mat1);

    auto mat2 = new lambertian(new solid_texture(vec3(0.4, 0.2, 0.1)));
    gpu_list[idx++] = new sphere(vec3(-4, 1, 0), 1, mat2);

    auto mat3 = new metal(vec3(0.7, 0.6, 0.5), 0);
    gpu_list[idx++] = new sphere(vec3(4, 1, 0), 1, mat3);
}

__global__ void create_world(hittable **gpu_list, hittable **gpu_world, camera **gpu_cam) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *gpu_world = new hittable_list(gpu_list, world_size);

        // init camera
        vec3 look_from(13, 2, 3);
        vec3 look_at(0, 0, 0);
        vec3 up(0, 1, 0);
        auto focus_dist = 10.0f;
        auto aperture = 0.1f;
        *gpu_cam = new camera(look_from, look_at, up, 90, float(nx)/ny, aperture, focus_dist);
    }
}
__global__ void free_world(hittable **gpu_list, hittable **gpu_world, camera **gpu_camera);

__global__ void random_state_init(curandState *rand_state);


int main() {

    cout << "Rendering a " << nx << "x" << ny << " image ";
    cout << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    // allocate buffer
    vec3 *fb = nullptr;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random states
    curandState *gpu_rand_states;
    checkCudaErrors(cudaMalloc((void**)&gpu_rand_states, num_pixels*sizeof(curandState)));

    // allocate world
    hittable **gpu_list = nullptr;
    checkCudaErrors(cudaMalloc((void**)&gpu_list, world_size*sizeof(void*)));
    hittable **gpu_world = nullptr;
    checkCudaErrors(cudaMalloc((void**)&gpu_world, sizeof(void*)));
    camera **gpu_cam = nullptr;
    checkCudaErrors(cudaMalloc((void**)&gpu_cam, sizeof(void*)));
    random_scene<<<1, 1>>>(gpu_list);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    create_world<<<1, 1>>>(gpu_list, gpu_world, gpu_cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // build bvh_tree
    bvh_node *gpu_bvh_tree = nullptr;
    checkCudaErrors(cudaMalloc((void**)&gpu_bvh_tree, sizeof(bvh_node)*(2*world_size+1)));
    {
        // start checking time
        clock_t start = clock();
        construct_bvh_node<<<1, 1>>>((hittable_list **) gpu_world, gpu_bvh_tree);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        cout << "build tree took " << ((double)(clock() - start)) / CLOCKS_PER_SEC << " seconds\n";
    }

    // start checking time
    clock_t start, stop;
    start = clock();

    // running info
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    // init random state
    random_state_init<<<blocks, threads>>>(gpu_rand_states);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render
    int *gpu_stack = nullptr;
    checkCudaErrors(cudaMalloc((void**)&gpu_stack, sizeof(int) * num_pixels * bvh_stack_max_depth));
    render<<<blocks, threads>>>(fb, gpu_cam, gpu_bvh_tree, gpu_rand_states, gpu_stack);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(gpu_stack));
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    cout << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    saveAsPPM("output.ppm", fb, nx, ny);

    // cleanup
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(gpu_list, gpu_world, gpu_cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(gpu_list));
    checkCudaErrors(cudaFree(gpu_world));
    checkCudaErrors(cudaFree(gpu_cam));
    checkCudaErrors(cudaFree(gpu_rand_states));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}

void check_cuda(cudaError_t result, const char *func, const char *file, int line) {
    if (result) {
        cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
             file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void free_world(hittable **gpu_list, hittable **gpu_world, camera **gpu_camera) {
    for(int i=0; i<world_size; ++i) {
        ((sphere*)gpu_list[i])->freeMat();
        delete gpu_list[i];
    }
    delete *gpu_world;
    delete *gpu_camera;
}

__global__ void random_state_init(curandState *rand_state) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) return;
    auto pixel_index = j*nx + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

