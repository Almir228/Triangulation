#include "surface.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace minimal;
void check(bool condition, const char *text) {
    if (!condition)
        throw std::runtime_error(text);
}
void monotone(const Result &r) {
    check(r.converged, "solver did not converge");
    for (size_t i = 1; i < r.areas.size(); ++i)
        check(r.areas[i] <= r.areas[i - 1] + 1e-12, "area increased");
}
template <class F> void rejects(F f) {
    bool failed = false;
    try {
        f();
    } catch (std::exception const &) {
        failed = true;
    }
    check(failed, "invalid input accepted");
}
void planar() {
    std::vector<Vec3> p;
    for (int i = 0; i < 32; ++i) {
        double t = 2 * std::acos(-1.) * i / 32;
        p.push_back({std::cos(t), std::sin(t), 0});
    }
    auto m = triangulate(p);
    refine(m);
    refine(m);
    auto before = m.vertices;
    for (size_t i = 0; i < m.vertices.size(); ++i)
        if (!m.boundary[i])
            m.vertices[i] = m.vertices[i] + m.normal * .2;
    int observed = 0, final_observations = 0;
    double last_observed_area = std::numeric_limits<double>::infinity();
    auto result =
        minimize(m, 100, 1e-9, [&](const Mesh &state, int, double area, double, bool final) {
            ++observed;
            final_observations += final;
            check(area <= last_observed_area + 1e-12, "observer saw increasing area");
            check(std::abs(area - evaluate(state, false).area * state.scale * state.scale) < 1e-12,
                  "observer area does not match its mesh");
            last_observed_area = area;
        });
    monotone(result);
    check(observed > 1 && final_observations == 1, "iteration observer did not report evolution");
    check(std::abs(evaluate(m, false).area * m.scale * m.scale -
                   16 * std::sin(2 * std::acos(-1.) / 32)) < 1e-10,
          "incorrect planar area");
    for (size_t i = 0; i < m.vertices.size(); ++i) {
        check(std::abs(m.vertices[i].z) < 1e-7, "planar interior not flat");
        if (m.boundary[i])
            check(norm(m.vertices[i] - before[i]) == 0, "boundary moved");
    }
    for (auto &v : p) {
        v = {5 + 2 * v.x, 7 + 2 * v.y / std::sqrt(2.), 11 + 2 * v.y / std::sqrt(2.)};
    }
    auto rotated = triangulate(p);
    refine(rotated);
    monotone(minimize(rotated));
    check(std::abs(evaluate(rotated, false).area * rotated.scale * rotated.scale -
                   64 * std::sin(2 * std::acos(-1.) / 32)) < 1e-10,
          "rotation/scale invariance failed");
}
void derivatives() {
    auto m = triangulate({{1, 0, .3}, {0, 1, -.2}, {-1, 0, .3}, {0, -1, -.2}});
    refine(m);
    auto base = evaluate(m);
    double eps = 1e-6;
    for (size_t i = 0; i < m.vertices.size(); ++i) {
        auto plus = m, minus = m;
        plus.vertices[i] = plus.vertices[i] + m.normal * eps;
        minus.vertices[i] = minus.vertices[i] - m.normal * eps;
        auto a = evaluate(plus), b = evaluate(minus);
        check(std::abs((a.area - b.area) / (2 * eps) - base.gradient[i]) < 1e-7,
              "area gradient failed finite differences");
        std::vector<double> column(m.vertices.size());
        for (size_t t = 0; t < m.faces.size(); ++t)
            for (int j = 0; j < 3; ++j)
                if (m.faces[t][j] == int(i))
                    for (int k = 0; k < 3; ++k)
                        column[m.faces[t][k]] += base.hessian[t][3 * k + j];
        for (size_t j = 0; j < column.size(); ++j)
            check(std::abs((a.gradient[j] - b.gradient[j]) / (2 * eps) - column[j]) < 2e-6,
                  "Hessian failed finite differences");
    }
}
void spatial_derivatives() {
    auto m = triangulate({{1, 0, .3}, {0, 1, -.2}, {-1, 0, .3}, {0, -1, -.2}});
    refine(m);
    auto base = evaluate_spatial(m);
    const double eps = 1e-6;
    for (size_t i = 0; i < m.vertices.size(); ++i)
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            auto plus = m, minus = m;
            double *positive[] = {&plus.vertices[i].x, &plus.vertices[i].y, &plus.vertices[i].z};
            double *negative[] = {&minus.vertices[i].x, &minus.vertices[i].y, &minus.vertices[i].z};
            *positive[coordinate] += eps;
            *negative[coordinate] -= eps;
            double numeric =
                (evaluate_spatial(plus, false).area - evaluate_spatial(minus, false).area) /
                (2 * eps);
            double analytic[] = {base.gradient[i].x, base.gradient[i].y, base.gradient[i].z};
            check(std::abs(numeric - analytic[coordinate]) < 1e-7,
                  "spatial area gradient failed finite differences");
        }
}
void spatial_overhang() {
    auto path = std::filesystem::temp_directory_path() / "minimal-spatial-overhang.obj";
    std::ofstream out(path);
    out << "v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nv 2 0 0.8\n"
           "f 1 2 5\nf 2 3 5\nf 3 4 5\nf 4 1 5\n";
    out.close();
    rejects([&] { read_obj(path.string(), {}, true); });
    auto m = read_obj(path.string(), {}, false);
    std::filesystem::remove(path);
    auto before = m.vertices;
    auto result = minimize_spatial(m, 100, 1e-9);
    monotone(result);
    check(std::abs(result.areas.back() - 4) < 1e-10,
          "spatial mode did not flatten overhanging mesh");
    for (size_t i = 0; i < m.vertices.size(); ++i)
        if (m.boundary[i])
            check(norm(m.vertices[i] - before[i]) == 0, "spatial mode moved boundary");
}
void edge_flips() {
    Mesh mesh;
    mesh.vertices = {{0, 0, 0}, {4, 0, 0}, {3, 1, 0}, {0, 2, 0}};
    mesh.faces = {{0, 1, 3}, {1, 2, 3}};
    mesh.boundary.assign(4, true);
    double area_before = evaluate_spatial(mesh, false).area;
    auto quality_before = mesh_quality(mesh);
    int flips = improve_spatial_mesh(mesh, 3);
    auto quality_after = mesh_quality(mesh);
    check(flips == 1, "beneficial edge was not flipped exactly once");
    check(quality_after.minimum > quality_before.minimum,
          "edge flip did not improve minimum triangle quality");
    check(std::abs(evaluate_spatial(mesh, false).area - area_before) < 1e-12,
          "planar edge flip changed surface area");
    check((mesh.faces[0] == std::array<int, 3>{0, 1, 2} ||
           mesh.faces[1] == std::array<int, 3>{0, 1, 2}),
          "edge flip produced unexpected connectivity");
}
void tangential_smoothing() {
    Mesh mesh;
    mesh.vertices = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}, {.8, .8, 0}};
    mesh.faces = {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}};
    mesh.boundary = {true, true, true, true, false};
    auto before_vertices = mesh.vertices;
    auto before_quality = mesh_quality(mesh);
    double before_area = evaluate_spatial(mesh, false).area;
    int passes = smooth_spatial_mesh(mesh, 3);
    check(passes > 0, "tangential smoothing rejected a planar improvement");
    check(mesh_quality(mesh).minimum > before_quality.minimum,
          "tangential smoothing did not improve minimum quality");
    check(std::abs(evaluate_spatial(mesh, false).area - before_area) < 1e-12,
          "planar tangential smoothing changed area");
    for (size_t i = 0; i < 4; ++i)
        check(norm(mesh.vertices[i] - before_vertices[i]) == 0,
              "tangential smoothing moved boundary");
}
double scherk(int n) {
    auto path =
        std::filesystem::temp_directory_path() /
        ("minimal-scherk-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".obj");
    std::ofstream o(path);
    o << std::setprecision(17);
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) {
            double x = -.65 + 1.3 * i / n, y = -.65 + 1.3 * j / n;
            double z = std::log(std::cos(y) / std::cos(x));
            if (i > 0 && j > 0 && i < n && j < n)
                z += .15;
            o << "v " << x << ' ' << y << ' ' << z << '\n';
        }
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            int a = j * (n + 1) + i + 1, b = a + 1, d = a + n + 1, c = d + 1;
            o << "f " << a << ' ' << b << ' ' << c << "\nf " << a << ' ' << c << ' ' << d << '\n';
        }
    o.close();
    auto m = read_obj(path.string(), {0, 0, 1});
    std::filesystem::remove(path);
    auto before = m.vertices;
    auto result = minimize(m);
    monotone(result);
    double error = 0;
    for (size_t i = 0; i < m.vertices.size(); ++i) {
        auto p = m.origin + m.vertices[i] * m.scale;
        error = std::max(error, std::abs(p.z - std::log(std::cos(p.y) / std::cos(p.x))));
        if (m.boundary[i])
            check(norm(m.vertices[i] - before[i]) == 0, "OBJ boundary moved");
    }
    check(result.areas.back() < result.areas.front(), "nonplanar area did not decrease");
    return error;
}
int main() {
    try {
        planar();
        derivatives();
        spatial_derivatives();
        spatial_overhang();
        edge_flips();
        tangential_smoothing();
        double a = scherk(4), b = scherk(8), c = scherk(16);
        std::cout << "Scherk max errors: " << a << ", " << b << ", " << c << '\n';
        check(b < a * .8 && c < b * .8 && c < .002, "Scherk refinement failed");
        rejects([] { triangulate({{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}); });
        rejects([] { triangulate({{0, 0, 0}, {1, 1, 0}, {0, 1, 0}, {1, 0, 0}}); });
        rejects([] { triangulate({{0, 0, 0}, {2, 0, 0}, {1, .5, 0}, {2, 2, 0}, {0, 2, 0}}); });
        rejects([] { triangulate({{0, 0, 0}, {1, 0, 0}, {1, 0, 0}, {0, 1, 0}}); });
        auto m = triangulate({{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, 0, 0}});
        check(m.boundary.size() == 4, "repeated closing point not handled");
        std::cout << "All numerical checks passed\n";
        return 0;
    } catch (std::exception const &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
