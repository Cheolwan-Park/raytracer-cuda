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

constexpr unsigned int nx = 640;
constexpr unsigned int ny = 360;
constexpr unsigned int ns = 128;
constexpr unsigned int nb = 50;
constexpr unsigned int tx = 16;
constexpr unsigned int ty = 16;

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
void check_cuda(cudaError_t result, const char *func, const char *file, int line);
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

__device__ vec3 color(const ray& r, hittable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation(1, 1, 1);
    for(int b=0; b < nb; ++b) {
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

__global__ void render(vec3 *fb, camera **cam, hittable **world, curandState *rand_state) {
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

constexpr int half_grid_len = 5;
constexpr unsigned int world_size = 4 + 4*half_grid_len*half_grid_len;
__global__ void create_world(hittable **gpu_list, hittable **gpu_world, camera **gpu_cam) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        gpu_list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
        gpu_list[1] = new sphere(vec3(0, 1, 0), 1, new dielectric(vec3(1, 1, 1), 1.5));
        gpu_list[2] = new sphere(vec3(-2.2, 1, 0), 1, new lambertian(vec3(0.4, 0.2, 0.1)));
        gpu_list[3] = new sphere(vec3(2.2, 1, 0), 1, new metal(vec3(0.7, 0.6, 0.5), 0.0));


        curandState rand_state;
        curand_init(1984, 0, 0, &rand_state);
        int i=4;
        for(int a=-half_grid_len; a<half_grid_len; ++a) {
            for(int b=-half_grid_len; b<half_grid_len; ++b) {
                auto choose_mat = curand_uniform(&rand_state);
                vec3 pos;
                do {
                    pos = vec3(float(a) + 0.9f * curand_uniform(&rand_state), 0.2 + 0.5f * curand_uniform(&rand_state),
                               float(b) + 0.9f * curand_uniform(&rand_state));
                } while((pos - vec3(4, 0.2, 0)).length() < 1.2f);
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
            }
        }

        *gpu_world = new hittable_list(gpu_list, world_size);

        // init camera
        vec3 look_from(0, 1, -3);
        vec3 look_at(0, 1, 0);
        vec3 up(0, 1, 0);
        auto focus_dist = 3.0f;
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

