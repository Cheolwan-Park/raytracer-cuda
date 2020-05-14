#ifndef RAYTRACER_STUDY_CAMERA_CUH
#define RAYTRACER_STUDY_CAMERA_CUH

#include "util.cuh"
#include "ray.cuh"

class camera {
public:
    __device__ explicit camera(const vec3 &look_from, const vec3 &look_at, const vec3 &up,
                                        float vertical_fov, float aspect_ratio, float aperture, float focus_dist)
    : _origin(look_from), _lower_left_corner(), _horizontal(), _vertical(), _u(), _v(), _w(){
        _lens_radius = aperture/2;

        auto theta = degrees_to_radians(vertical_fov);
        auto half_height = tan(theta/2);
        auto half_width = aspect_ratio * half_height;

        _w = unit_vector(look_from - look_at);
        _u = unit_vector(cross(up, _w));
        _v = cross(_w, _u);

        _lower_left_corner = _origin
                            - half_width*focus_dist*_u
                            - half_height*focus_dist*_v
                            - focus_dist*_w;
        _horizontal = 2*half_width*focus_dist*_u;
        _vertical = 2*half_height*focus_dist*_v;
    }

    __device__ ray getRay(float s, float t, curandState *rand_state) {
        vec3 rd = _lens_radius * random_in_unit_disk(rand_state);
        vec3 offset = rd.x()*_u + rd.y()*_v;
        return ray(_origin + offset, _lower_left_corner + s*_horizontal + t*_vertical - _origin - offset);
    }

private:
    vec3 _origin;
    vec3 _lower_left_corner;
    vec3 _horizontal;
    vec3 _vertical;
    vec3 _u, _v, _w;
    float _lens_radius;
};

#endif //RAYTRACER_STUDY_CAMERA_CUH
