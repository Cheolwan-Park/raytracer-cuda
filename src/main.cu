#include <iostream>
#include <ctime>
#include "vec3.h"
#include "ray.h"
#include "material.h"
#include "camera.h"
#include "hittables/hittable_list.h"
#include "hittables/sphere.h"

using namespace std;

constexpr unsigned int nx = 1920;
constexpr unsigned int ny = 1080;
constexpr unsigned int ns = 128;
constexpr unsigned int tx = 16;
constexpr unsigned int ty = 16;

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
void check_cuda(cudaError_t result, const char *func, const char *file, int line);
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

__device__ vec3 color(const ray& r, hittable_list **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation(1, 1, 1);
    for(int bounce=0; bounce<50; ++bounce) {
        hit_record rec;
        if((*world)->hit(cur_ray, 0.0001f, infinity, rec)) {
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

__global__ void render(vec3 *fb, camera **cam, hittable_list **world, curandState *rand_state) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) return;

    auto pixel_index = j*nx + i;
    curandState local_rand_state = rand_state[pixel_index];

    vec3 col(0, 0, 0);
    for(int s=0; s<ns; ++s) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(nx);
        float v = float(j + curand_uniform(&local_rand_state)) / float(ny);
        ray r = (*cam)->getRay(u, v, rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col = sqrt(col);
    fb[pixel_index] = col;
}

constexpr unsigned int world_size = 22*22+4;
__global__ void create_world(hittable **gpu_list, hittable_list **gpu_world, camera **gpu_cam) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState rand_state;

        gpu_list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
        gpu_list[1] = new sphere(vec3(0, 1, 0), 1, new dielectric(vec3(1, 1, 1), 1.5));
        gpu_list[2] = new sphere(vec3(-4, 1, 0), 1, new lambertian(vec3(0.4, 0.2, 0.1)));
        gpu_list[3] = new sphere(vec3(4, 1, 0), 1, new metal(vec3(0.7, 0.6, 0.5), 0.0));

        int i=4;
        for(int a=-11; a<11; ++a) {
            for(int b=-11; b<11; ++b) {
                auto choose_mat = curand_uniform(&rand_state);
                vec3 pos(float(a)+0.9f*curand_uniform(&rand_state), 0.2, float(b)+0.9f*curand_uniform(&rand_state));
                if((pos - vec3(4, 0.2, 0)).length() > 0.9f) {
                    if(choose_mat < 0.8f) {
                        // diffuse
                        auto albedo = random01_vec3(&rand_state) * random01_vec3(&rand_state);
                        gpu_list[i++] = new sphere(pos, 0.2, new lambertian(albedo));
                    } else if(choose_mat < 0.95f) {
                        // metal
                        auto albedo = random_vec3(&rand_state, 0.5f, 1.0f);
                        auto fuzz = gpu_random(&rand_state, 0, 0.5f);
                        gpu_list[i++] = new sphere(pos, 0.2, new metal(albedo, fuzz));
                    } else {
                        // glass
                        auto albedo = random_vec3(&rand_state, 0.5f, 1.0f);
                        gpu_list[i++] = new sphere(pos, 0.2, new dielectric(albedo, 1.5f));
                    }
                } else {
                    gpu_list[i++] = nullptr;
                }
            }
        }

        *gpu_world = new hittable_list(gpu_list, world_size);

        // init camera
        vec3 look_from(13, 2, 3);
        vec3 look_at(0, 0, 0);
        vec3 up(0, 1, 0);
        auto focus_dist = 10.0f;
        auto aperture = 0.1f;
        *gpu_cam = new camera(look_from, look_at, up, 20, float(nx)/ny, aperture, focus_dist);
    }
}
__global__ void free_world(hittable **gpu_list, hittable_list **gpu_world, camera **gpu_camera);

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
    hittable_list **gpu_world = nullptr;
    checkCudaErrors(cudaMalloc((void**)&gpu_world, sizeof(void*)));
    camera **gpu_cam = nullptr;
    checkCudaErrors(cudaMalloc((void**)&gpu_cam, sizeof(void*)));
    create_world<<<1, 1>>>(gpu_list, gpu_world, gpu_cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

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
    render<<<blocks, threads>>>(fb, gpu_cam, gpu_world, gpu_rand_states);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
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

__global__ void free_world(hittable **gpu_list, hittable_list **gpu_world, camera **gpu_camera) {
    for(int i=0; i<world_size; ++i) {
        if(!gpu_list[i]) continue;
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

