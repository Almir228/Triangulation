#include "surface.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace minimal {
namespace {
using Edge = std::pair<int, int>;
struct EdgeUse {
    size_t face = 0;
    int from = 0, to = 0, opposite = 0;
};

double triangle_quality(Vec3 a, Vec3 b, Vec3 c) {
    double twice_area = norm(cross(b - a, c - a));
    double squared_edges = dot(b - a, b - a) + dot(c - b, c - b) + dot(a - c, a - c);
    if (!(twice_area > 0) || !(squared_edges > 0))
        return 0;
    return 2 * std::sqrt(3.0) * twice_area / squared_edges;
}

double triangle_area(Vec3 a, Vec3 b, Vec3 c) {
    return 0.5 * norm(cross(b - a, c - a));
}
} // namespace

MeshQuality mesh_quality(const Mesh &mesh) {
    if (mesh.faces.empty())
        throw std::runtime_error("Нельзя оценить качество пустой сетки.");
    MeshQuality quality{1, 0};
    for (auto face : mesh.faces) {
        double value = triangle_quality(mesh.vertices[face[0]], mesh.vertices[face[1]],
                                        mesh.vertices[face[2]]);
        quality.minimum = std::min(quality.minimum, value);
        quality.mean += value / mesh.faces.size();
    }
    return quality;
}

int improve_spatial_mesh(Mesh &mesh, int passes) {
    if (passes < 0 || passes > 100)
        throw std::runtime_error("Число проходов перестройки должно быть от 0 до 100.");
    int total_flips = 0;
    for (int pass = 0; pass < passes; ++pass) {
        std::map<Edge, std::vector<EdgeUse>> uses;
        std::vector<std::set<int>> neighbors(mesh.vertices.size());
        for (size_t face_id = 0; face_id < mesh.faces.size(); ++face_id) {
            auto face = mesh.faces[face_id];
            for (int i = 0; i < 3; ++i) {
                int a = face[i], b = face[(i + 1) % 3], c = face[(i + 2) % 3];
                uses[std::minmax(a, b)].push_back({face_id, a, b, c});
                neighbors[a].insert(b);
                neighbors[b].insert(a);
            }
        }
        std::set<size_t> changed_faces;
        std::set<int> changed_vertices;
        int pass_flips = 0;
        for (auto const &[edge, adjacent] : uses) {
            (void)edge;
            if (adjacent.size() != 2)
                continue;
            auto first = adjacent[0], second = adjacent[1];
            if (first.from != second.to || first.to != second.from ||
                changed_faces.count(first.face) || changed_faces.count(second.face))
                continue;
            int a = first.from, b = first.to, c = first.opposite, d = second.opposite;
            if (changed_vertices.count(a) || changed_vertices.count(b) ||
                changed_vertices.count(c) || changed_vertices.count(d))
                continue;
            if (c == d || uses.count(std::minmax(c, d)))
                continue;
            if ((!mesh.boundary[a] && neighbors[a].size() <= 3) ||
                (!mesh.boundary[b] && neighbors[b].size() <= 3))
                continue;
            Vec3 pa = mesh.vertices[a], pb = mesh.vertices[b], pc = mesh.vertices[c],
                 pd = mesh.vertices[d];
            double before_quality =
                std::min(triangle_quality(pa, pb, pc), triangle_quality(pb, pa, pd));
            Vec3 normal_sum = cross(pb - pa, pc - pa) + cross(pa - pb, pd - pb);
            Vec3 first_normal = cross(pa - pc, pd - pc);
            Vec3 second_normal = cross(pd - pc, pb - pc);
            double normal_scale = norm(normal_sum);
            if (!(norm(first_normal) > 1e-14) || !(norm(second_normal) > 1e-14) ||
                !(normal_scale > 1e-14) ||
                dot(first_normal, normal_sum) <= 1e-12 * norm(first_normal) * normal_scale ||
                dot(second_normal, normal_sum) <= 1e-12 * norm(second_normal) * normal_scale)
                continue;
            double after_quality =
                std::min(triangle_quality(pc, pa, pd), triangle_quality(pc, pd, pb));
            if (after_quality <= before_quality + 1e-12)
                continue;
            double before_area = triangle_area(pa, pb, pc) + triangle_area(pb, pa, pd);
            double after_area = triangle_area(pc, pa, pd) + triangle_area(pc, pd, pb);
            if (after_area > before_area * (1 + 1e-12))
                continue;
            mesh.faces[first.face] = {c, a, d};
            mesh.faces[second.face] = {c, d, b};
            changed_faces.insert(first.face);
            changed_faces.insert(second.face);
            changed_vertices.insert({a, b, c, d});
            ++pass_flips;
        }
        total_flips += pass_flips;
        if (!pass_flips)
            break;
    }
    return total_flips;
}

int smooth_spatial_mesh(Mesh &mesh, int passes, double strength) {
    if (passes < 0 || passes > 100 || !(strength > 0 && strength <= 1))
        throw std::runtime_error("Некорректные параметры сглаживания сетки.");
    int accepted_passes = 0;
    for (int pass = 0; pass < passes; ++pass) {
        std::vector<std::set<int>> neighbors(mesh.vertices.size());
        std::vector<Vec3> normals(mesh.vertices.size());
        for (auto face : mesh.faces) {
            Vec3 face_normal = cross(mesh.vertices[face[1]] - mesh.vertices[face[0]],
                                     mesh.vertices[face[2]] - mesh.vertices[face[0]]);
            for (int i = 0; i < 3; ++i) {
                normals[face[i]] = normals[face[i]] + face_normal;
                neighbors[face[i]].insert(face[(i + 1) % 3]);
                neighbors[face[i]].insert(face[(i + 2) % 3]);
            }
        }
        std::vector<Vec3> displacement(mesh.vertices.size());
        double largest = 0;
        for (size_t i = 0; i < mesh.vertices.size(); ++i) {
            if (mesh.boundary[i] || neighbors[i].empty() || norm(normals[i]) < 1e-14)
                continue;
            Vec3 centroid;
            for (int neighbor : neighbors[i])
                centroid = centroid + mesh.vertices[neighbor] * (1.0 / neighbors[i].size());
            Vec3 unit_normal = normals[i] * (1 / norm(normals[i]));
            Vec3 toward_centroid = centroid - mesh.vertices[i];
            displacement[i] =
                (toward_centroid - unit_normal * dot(toward_centroid, unit_normal)) * strength;
            largest = std::max(largest, norm(displacement[i]));
        }
        if (!(largest > 1e-14))
            break;
        auto original = mesh.vertices;
        auto before_quality = mesh_quality(mesh);
        double before_area = evaluate_spatial(mesh, false).area;
        bool accepted = false;
        for (double step = 1; step >= 1.0 / 256; step *= 0.5) {
            for (size_t i = 0; i < mesh.vertices.size(); ++i)
                if (!mesh.boundary[i])
                    mesh.vertices[i] = original[i] + displacement[i] * step;
            try {
                auto after_quality = mesh_quality(mesh);
                double after_area = evaluate_spatial(mesh, false).area;
                bool quality_improved =
                    after_quality.minimum >= before_quality.minimum * (1 - 1e-10) &&
                    after_quality.mean > before_quality.mean + 1e-12;
                if (quality_improved && after_area <= before_area * (1 + 1e-8)) {
                    accepted = true;
                    break;
                }
            } catch (const std::runtime_error &) {
            }
        }
        if (!accepted) {
            mesh.vertices = std::move(original);
            break;
        }
        ++accepted_passes;
    }
    return accepted_passes;
}
} // namespace minimal
