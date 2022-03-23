#ifndef RAYTRACER_STUDY_TEXTURE_CUH
#define RAYTRACER_STUDY_TEXTURE_CUH

#include "util.cuh"
#include <stdio.h>
#include <memory>

class base_texture {
public:
    __device__ virtual ~base_texture() { ; }
    __device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class solid_texture : public base_texture {
public:
    __device__ solid_texture() : _color() { ; }
    __device__ explicit solid_texture(const vec3 &col) : _color(col) { ; }
    __device__ explicit solid_texture(float r, float g, float b) : _color(r, g, b) { ; }

    __device__ vec3 value(float u, float v, const vec3& p) const override {
        return _color;
    }

private:
    vec3 _color;
};

class checker_texture : public base_texture {
public:
    __device__ checker_texture() = delete;
    __device__ explicit checker_texture(base_texture *odd, base_texture *even) : _tex{odd, even} { ; }

    __device__ ~checker_texture() override {
        delete _tex[0];
        delete _tex[1];
    }

    __device__ vec3 value(float u, float v, const vec3& p) const override {
        auto sine = sin(10*p.x())*sin(10*p.y())*sin(10*p.z());
        return _tex[int(ceil(sine))]->value(u, v, p);
    }

private:
    base_texture *_tex[2];
};


#endif //RAYTRACER_STUDY_TEXTURE_CUH
