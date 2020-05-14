#ifndef RAYTRACER_STUDY_MATERIAL_CUH
#define RAYTRACER_STUDY_MATERIAL_CUH

#include "util.cuh"
#include "ray.cuh"
#include "hittables/hittable.cuh"

class material {
public:
    __device__ virtual ~material() { ; }
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation,
                                            ray& scattered, curandState *rand_state) const = 0;
};

class lambertian : public material {
public:
    __device__ explicit lambertian(const vec3 &albedo) : _albedo(albedo) { ; }

    __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation,
                                      ray& scattered, curandState *rand_state) const override {
        vec3 tgt_dir = rec.normal + random_unit_vector(rand_state);
        scattered = ray(rec.p, tgt_dir);
        attenuation = _albedo;
        return true;
    }

    // get
    __device__ const vec3 &albedo() const { return _albedo; }
    __device__ vec3 &albedo() { return _albedo; }

    // set
    __device__ void setAlbedo(const vec3 &v) { _albedo = v; }

private:
    vec3 _albedo;
};

class metal : public material {
public:
    __device__ explicit metal(const vec3& a, float f) : _albedo(a), _fuzz(f > 1 ? 1 : f) { ; }

    __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation,
                                    ray& scattered, curandState *rand_state) const override {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + _fuzz*random_in_unit_sphere(rand_state));
        attenuation = _albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }

    // get
    __device__ const vec3 &albedo() const { return _albedo; }
    __device__ vec3 &albedo() { return _albedo; }

    __device__ float getFuzz() const { return _fuzz; }

    // set
    __device__ void setAlbedo(const vec3 &v) { _albedo = v; }

    __device__ void setFuzz(float f) { _fuzz = f; }

private:
    vec3 _albedo;
    float _fuzz;
};

class dielectric : public material {
public:
    __device__ explicit dielectric(const vec3 &a, float ri) : _albedo(a), _ref_idx(ri) { ; }

    __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation,
                                      ray& scattered, curandState *rand_state) const override {
        attenuation = _albedo;

        vec3 outward_normal;
        vec3 unit_dir = unit_vector(r_in.direction());
        float ni_over_nt = 0.0f;
        vec3 refracted;
        float reflect_prob;
        float cosine;

        if (dot(unit_dir, rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = _ref_idx;
            cosine = dot(unit_dir, rec.normal);
            cosine = sqrt(1.0f - _ref_idx*_ref_idx*(1-cosine*cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / _ref_idx;
            cosine = -dot(unit_dir, rec.normal);
        }

        if (refract(unit_dir, outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, _ref_idx);
        else
            reflect_prob = 1.0f;

        if (curand_uniform(rand_state) < reflect_prob)
            scattered = ray(rec.p, reflect(r_in.direction(), rec.normal));
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

private:
    vec3 _albedo;
    float _ref_idx;
};

#endif //RAYTRACER_STUDY_MATERIAL_CUH
