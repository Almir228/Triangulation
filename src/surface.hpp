#pragma once
#include <array>
#include <functional>
#include <string>
#include <vector>

namespace minimal {
struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3 operator+(Vec3 b) const { return {x + b.x, y + b.y, z + b.z}; }
    Vec3 operator-(Vec3 b) const { return {x - b.x, y - b.y, z - b.z}; }
    Vec3 operator*(double a) const { return {x * a, y * a, z * a}; }
};
double dot(Vec3 a, Vec3 b);
Vec3 cross(Vec3 a, Vec3 b);
double norm(Vec3 a);
struct Mesh {
    std::vector<Vec3> vertices; // Нормированные координаты, начало в origin.
    std::vector<std::array<int, 3>> faces;
    std::vector<bool> boundary;
    Vec3 normal, origin;
    double scale = 1;
};
struct Evaluation {
    double area = 0;
    std::vector<double> gradient;
    std::vector<std::array<double, 9>> hessian;
};
struct SpatialEvaluation {
    double area = 0;
    double minimum_twice_area = 0;
    std::vector<Vec3> gradient;
};
struct MeshQuality {
    double minimum = 0;
    double mean = 0;
};
struct Result {
    bool converged = false;
    int iterations = 0;
    double residual = 0;
    std::vector<double> areas;
};
struct AnimationTopology {
    std::vector<std::array<int, 3>> faces;
};
struct AnimationFrame {
    std::vector<Vec3> vertices;
    size_t topology = 0;
    double area = 0;
    double residual = 0;
    int level = 0;
    int iteration = 0;
    double quality = 0;
};
using IterationObserver =
    std::function<void(const Mesh &, int iteration, double area, double residual, bool final)>;
std::vector<Vec3> read_contour(const std::string &path);
Mesh triangulate(std::vector<Vec3> contour, Vec3 normal = {});
Mesh read_obj(const std::string &path, Vec3 normal = {}, bool require_graph = true);
void refine(Mesh &mesh);
MeshQuality mesh_quality(const Mesh &mesh);
int improve_spatial_mesh(Mesh &mesh, int passes = 1);
int smooth_spatial_mesh(Mesh &mesh, int passes = 1, double strength = 0.35);
Evaluation evaluate(const Mesh &mesh, bool derivatives = true);
SpatialEvaluation evaluate_spatial(const Mesh &mesh, bool derivatives = true);
Result minimize(Mesh &mesh, int max_iterations = 100, double tolerance = 1e-9,
                const IterationObserver &observer = {});
Result minimize_spatial(Mesh &mesh, int max_iterations = 100, double tolerance = 1e-9,
                        const IterationObserver &observer = {});
void write_obj(const Mesh &mesh, const std::string &path);
void write_html(const Mesh &mesh, const std::string &path,
                const std::vector<AnimationTopology> &topologies,
                const std::vector<AnimationFrame> &frames);
} // namespace minimal
