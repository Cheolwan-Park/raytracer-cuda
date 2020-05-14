#ifndef RAYTRACER_STUDY_VEC3_CUH
#define RAYTRACER_STUDY_VEC3_CUH

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <cmath>
#include <cstdlib>
#include <iostream>

class vec3  {
public:
    CUDA_CALLABLE_MEMBER vec3(): e{0.0f, 0.0f, 0.0f} { ; }
    CUDA_CALLABLE_MEMBER explicit vec3(float e0, float e1, float e2): e{e0, e1, e2} { ; }
    CUDA_CALLABLE_MEMBER inline float x() const { return e[0]; }
    CUDA_CALLABLE_MEMBER inline float y() const { return e[1]; }
    CUDA_CALLABLE_MEMBER inline float z() const { return e[2]; }
    CUDA_CALLABLE_MEMBER inline float r() const { return e[0]; }
    CUDA_CALLABLE_MEMBER inline float g() const { return e[1]; }
    CUDA_CALLABLE_MEMBER inline float b() const { return e[2]; }

    CUDA_CALLABLE_MEMBER inline const vec3& operator+() const { return *this; }
    CUDA_CALLABLE_MEMBER inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    CUDA_CALLABLE_MEMBER inline float operator[](int i) const { return e[i]; }
    CUDA_CALLABLE_MEMBER inline float& operator[](int i) { return e[i]; };

    CUDA_CALLABLE_MEMBER inline vec3& operator+=(const vec3 &v2);
    CUDA_CALLABLE_MEMBER inline vec3& operator-=(const vec3 &v2);
    CUDA_CALLABLE_MEMBER inline vec3& operator*=(const vec3 &v2);
    CUDA_CALLABLE_MEMBER inline vec3& operator/=(const vec3 &v2);
    CUDA_CALLABLE_MEMBER inline vec3& operator*=(float t);
    CUDA_CALLABLE_MEMBER inline vec3& operator/=(float t);

    CUDA_CALLABLE_MEMBER inline float length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
    CUDA_CALLABLE_MEMBER inline float squared_length() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
    CUDA_CALLABLE_MEMBER inline void make_unit_vector();

private:
    float e[3];
};

inline std::istream& operator>>(std::istream &is, vec3 &t) {
    is >> t[0] >> t[1] >> t[2];
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &t) {
    os << t[0] << " " << t[1] << " " << t[2];
    return os;
}

CUDA_CALLABLE_MEMBER inline void vec3::make_unit_vector() {
    float k = 1.0f / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}

CUDA_CALLABLE_MEMBER inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
    return vec3(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);
}

CUDA_CALLABLE_MEMBER inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
    return vec3(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]);
}

CUDA_CALLABLE_MEMBER inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
    return vec3(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]);
}

CUDA_CALLABLE_MEMBER inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
    return vec3(v1[0] / v2[0], v1[1] / v2[1], v1[2] / v2[2]);
}

CUDA_CALLABLE_MEMBER inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t*v[0], t*v[1], t*v[2]);
}

CUDA_CALLABLE_MEMBER inline vec3 operator/(vec3 v, float t) {
    return vec3(v[0]/t, v[1]/t, v[2]/t);
}

CUDA_CALLABLE_MEMBER inline vec3 operator*(const vec3 &v, float t) {
    return vec3(t*v[0], t*v[1], t*v[2]);
}

CUDA_CALLABLE_MEMBER inline float dot(const vec3 &v1, const vec3 &v2) {
    return v1[0] *v2[0] + v1[1] *v2[1]  + v1[2] *v2[2];
}

CUDA_CALLABLE_MEMBER inline vec3 cross(const vec3 &v1, const vec3 &v2) {
    return vec3( (v1[1]*v2[2] - v1[2]*v2[1]),
                 (-(v1[0]*v2[2] - v1[2]*v2[0])),
                 (v1[0]*v2[1] - v1[1]*v2[0]));
}

CUDA_CALLABLE_MEMBER inline vec3& vec3::operator+=(const vec3 &v){
    e[0]  += v[0];
    e[1]  += v[1];
    e[2]  += v[2];
    return *this;
}

CUDA_CALLABLE_MEMBER inline vec3& vec3::operator*=(const vec3 &v){
    e[0]  *= v[0];
    e[1]  *= v[1];
    e[2]  *= v[2];
    return *this;
}

CUDA_CALLABLE_MEMBER inline vec3& vec3::operator/=(const vec3 &v){
    e[0]  /= v[0];
    e[1]  /= v[1];
    e[2]  /= v[2];
    return *this;
}

CUDA_CALLABLE_MEMBER inline vec3& vec3::operator-=(const vec3& v) {
    e[0]  -= v[0];
    e[1]  -= v[1];
    e[2]  -= v[2];
    return *this;
}

CUDA_CALLABLE_MEMBER inline vec3& vec3::operator*=(float t) {
    e[0]  *= t;
    e[1]  *= t;
    e[2]  *= t;
    return *this;
}

CUDA_CALLABLE_MEMBER inline vec3& vec3::operator/=(float t) {
    float k = 1.0f/t;

    e[0]  *= k;
    e[1]  *= k;
    e[2]  *= k;
    return *this;
}

CUDA_CALLABLE_MEMBER inline vec3 unit_vector(const vec3 &v) {
    return v / v.length();
}

CUDA_CALLABLE_MEMBER inline vec3 sqrt(vec3 v) {
    v[0] = sqrt(v[0]);
    v[1] = sqrt(v[1]);
    v[2] = sqrt(v[2]);
    return v;
}

#endif