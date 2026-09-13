#include "surface.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace minimal {
double dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double norm(Vec3 a) {
    return std::sqrt(dot(a, a));
}
std::vector<Vec3> read_contour(const std::string &path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Не удалось открыть контур: " + path);
    std::vector<Vec3> points;
    std::string line;
    int number = 0;
    while (std::getline(in, line)) {
        ++number;
        line = line.substr(0, line.find('#'));
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        row >> std::ws;
        if (row.eof())
            continue;
        Vec3 p;
        std::string extra;
        if (!(row >> p.x >> p.y >> p.z) || (row >> extra) || !std::isfinite(p.x) ||
            !std::isfinite(p.y) || !std::isfinite(p.z))
            throw std::runtime_error("Ожидались три конечные координаты, строка " +
                                     std::to_string(number));
        points.push_back(p);
    }
    return points;
}
Mesh triangulate(std::vector<Vec3> p, Vec3 preferred) {
    if (p.size() > 1 && norm(p.front() - p.back()) == 0)
        p.pop_back();
    if (p.size() < 3 || p.size() > 4096)
        throw std::runtime_error("Нужно от 3 до 4096 точек контура.");
    Mesh m;
    m.origin = p.front();
    // Сначала сдвиг, затем нормирование: устойчивость к масштабу и переносу.
    double scale = 0;
    for (auto &v : p) {
        if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
            throw std::runtime_error("Неконечная координата.");
        v = v - m.origin;
        scale = std::max(scale, norm(v));
    }
    if (!(scale > 0) || !std::isfinite(scale))
        throw std::runtime_error("Вырожденный масштаб контура.");
    m.scale = scale;
    Vec3 center;
    for (auto &v : p) {
        v = v * (1 / scale);
        center = center + v * (1.0 / p.size());
    }
    Vec3 n;
    for (size_t i = 0; i < p.size(); ++i)
        n = n + cross(p[i] - center, p[(i + 1) % p.size()] - center);
    if (norm(n) < 1e-12)
        throw std::runtime_error(
            "Не удалось определить плоскость проекции: контур вырожден или самопересекается.");
    m.normal = n * (1 / norm(n));
    if (norm(preferred) > 0) {
        if (!std::isfinite(norm(preferred)))
            throw std::runtime_error("Некорректная нормаль.");
        m.normal = preferred * (1 / norm(preferred));
        if (dot(n, m.normal) < 0)
            std::reverse(p.begin(), p.end());
    }
    // Глобальная проверка полуплоскостей отвергает звёздчатые самопересечения,
    // которые нельзя распознать только по знакам поворота соседних рёбер.
    for (size_t i = 0; i < p.size(); ++i) {
        Vec3 e = p[(i + 1) % p.size()] - p[i];
        if (norm(cross(e, m.normal)) < 1e-10)
            throw std::runtime_error("Повторные точки или нулевая длина ребра в проекции.");
        for (size_t j = 0; j < p.size(); ++j) {
            if (j == i || j == (i + 1) % p.size())
                continue;
            if (dot(cross(e, p[j] - p[i]), m.normal) < -1e-12)
                throw std::runtime_error(
                    "Нужен простой контур с выпуклой проекцией; проверьте порядок точек.");
            if (norm(p[j] - p[i]) < 1e-12)
                throw std::runtime_error("Повторная точка контура.");
        }
        if (dot(cross(e, center - p[i]), m.normal) < 1e-12)
            throw std::runtime_error("Вырожденная проекция контура.");
    }
    m.vertices = p;
    m.boundary.assign(p.size(), true);
    m.vertices.push_back(center);
    m.boundary.push_back(false);
    for (size_t i = 0; i < p.size(); ++i)
        m.faces.push_back({int(i), int((i + 1) % p.size()), int(p.size())});
    return m;
}
void refine(Mesh &m) {
    if (m.faces.size() > 50000)
        throw std::runtime_error("Сгущение превысит предел 200000 треугольников.");
    using Edge = std::pair<int, int>;
    std::map<Edge, int> counts, mid;
    for (auto f : m.faces)
        for (int i = 0; i < 3; ++i)
            ++counts[std::minmax(f[i], f[(i + 1) % 3])];
    auto midpoint = [&](int a, int b) {
        Edge key = std::minmax(a, b);
        auto it = mid.find(key);
        if (it != mid.end())
            return it->second;
        int id = int(m.vertices.size());
        mid[key] = id;
        m.vertices.push_back((m.vertices[a] + m.vertices[b]) * 0.5);
        m.boundary.push_back(counts.at(key) == 1);
        return id;
    };
    std::vector<std::array<int, 3>> faces;
    faces.reserve(m.faces.size() * 4);
    for (auto f : m.faces) {
        int a = midpoint(f[0], f[1]), b = midpoint(f[1], f[2]), c = midpoint(f[2], f[0]);
        faces.push_back({f[0], a, c});
        faces.push_back({a, f[1], b});
        faces.push_back({c, b, f[2]});
        faces.push_back({a, b, c});
    }
    m.faces = std::move(faces);
}
Evaluation evaluate(const Mesh &m, bool derivatives) {
    Evaluation out;
    if (derivatives) {
        out.gradient.resize(m.vertices.size());
        out.hessian.reserve(m.faces.size());
    }
    for (auto f : m.faces) {
        Vec3 u = m.vertices[f[1]] - m.vertices[f[0]], v = m.vertices[f[2]] - m.vertices[f[0]],
             c = cross(u, v);
        double length = norm(c);
        if (!(length > 1e-18) || !std::isfinite(length))
            throw std::runtime_error("Вырожденный треугольник.");
        out.area += length * 0.5;
        if (!derivatives)
            continue;
        Vec3 dc[3] = {cross(v - u, m.normal), cross(m.normal, v), cross(u, m.normal)};
        std::array<double, 9> h{};
        for (int i = 0; i < 3; ++i) {
            out.gradient[f[i]] += 0.5 * dot(c, dc[i]) / length;
            for (int j = 0; j < 3; ++j)
                h[3 * i + j] = 0.5 * (dot(dc[i], dc[j]) / length -
                                      dot(c, dc[i]) * dot(c, dc[j]) / (length * length * length));
        }
        out.hessian.push_back(h);
    }
    return out;
}
SpatialEvaluation evaluate_spatial(const Mesh &m, bool derivatives) {
    SpatialEvaluation out;
    out.minimum_twice_area = std::numeric_limits<double>::infinity();
    if (derivatives)
        out.gradient.resize(m.vertices.size());
    for (auto f : m.faces) {
        Vec3 u = m.vertices[f[1]] - m.vertices[f[0]];
        Vec3 v = m.vertices[f[2]] - m.vertices[f[0]];
        Vec3 area_vector = cross(u, v);
        double twice_area = norm(area_vector);
        if (!(twice_area > 1e-18) || !std::isfinite(twice_area))
            throw std::runtime_error("Вырожденный пространственный треугольник.");
        out.area += 0.5 * twice_area;
        out.minimum_twice_area = std::min(out.minimum_twice_area, twice_area);
        if (!derivatives)
            continue;
        Vec3 unit_normal = area_vector * (1 / twice_area);
        Vec3 g1 = cross(v, unit_normal) * 0.5;
        Vec3 g2 = cross(unit_normal, u) * 0.5;
        out.gradient[f[0]] = out.gradient[f[0]] + (g1 + g2) * -1;
        out.gradient[f[1]] = out.gradient[f[1]] + g1;
        out.gradient[f[2]] = out.gradient[f[2]] + g2;
    }
    return out;
}
static double inner(const std::vector<double> &a, const std::vector<double> &b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}
Result minimize(Mesh &m, int max_iterations, double tolerance, const IterationObserver &observer) {
    if (max_iterations < 1 || !(tolerance > 0) || !std::isfinite(tolerance))
        throw std::runtime_error("Некорректные параметры оптимизации.");
    Result result;
    for (int iteration = 0; iteration <= max_iterations; ++iteration) {
        auto e = evaluate(m);
        result.areas.push_back(e.area * m.scale * m.scale);
        result.iterations = iteration;
        std::vector<double> g = e.gradient;
        double residual = 0;
        for (size_t i = 0; i < g.size(); ++i) {
            if (m.boundary[i])
                g[i] = 0;
            residual = std::max(residual, std::abs(g[i]));
        }
        result.residual = residual;
        if (observer)
            observer(m, iteration, result.areas.back(), residual, false);
        if (residual <= tolerance) {
            result.converged = true;
            if (observer)
                observer(m, iteration, result.areas.back(), residual, true);
            return result;
        }
        if (iteration == max_iterations) {
            if (observer)
                observer(m, iteration, result.areas.back(), residual, true);
            return result;
        }
        const size_t n = g.size();
        std::vector<double> diagonal(n, 1e-12);
        for (size_t t = 0; t < m.faces.size(); ++t)
            for (int i = 0; i < 3; ++i)
                diagonal[m.faces[t][i]] += e.hessian[t][3 * i + i];
        auto multiply = [&](const std::vector<double> &x) {
            std::vector<double> y(n);
            for (size_t t = 0; t < m.faces.size(); ++t) {
                auto f = m.faces[t];
                for (int i = 0; i < 3; ++i)
                    if (!m.boundary[f[i]])
                        for (int j = 0; j < 3; ++j)
                            if (!m.boundary[f[j]])
                                y[f[i]] += e.hessian[t][3 * i + j] * x[f[j]];
            }
            for (size_t i = 0; i < n; ++i)
                y[i] += 1e-12 * x[i];
            return y;
        };
        // Приближённый шаг Ньютона: CG с диагональным предобуславливанием.
        std::vector<double> step(n), r(n), z(n), direction(n);
        for (size_t i = 0; i < n; ++i) {
            r[i] = -g[i];
            z[i] = r[i] / diagonal[i];
        }
        direction = z;
        double rz = inner(r, z), initial = inner(r, r);
        for (size_t k = 0; k < std::min<size_t>(2 * n, 2000) && inner(r, r) > initial * 1e-12;
             ++k) {
            auto hd = multiply(direction);
            double denom = inner(direction, hd);
            if (!(denom > 0))
                break;
            double alpha = rz / denom;
            for (size_t i = 0; i < n; ++i) {
                step[i] += alpha * direction[i];
                r[i] -= alpha * hd[i];
                z[i] = r[i] / diagonal[i];
            }
            double next = inner(r, z);
            if (!(rz > 0))
                break;
            for (size_t i = 0; i < n; ++i)
                direction[i] = z[i] + (next / rz) * direction[i];
            rz = next;
        }
        double descent = inner(g, step);
        if (!(descent < 0)) {
            for (size_t i = 0; i < n; ++i)
                step[i] = -g[i] / diagonal[i];
            descent = inner(g, step);
        }
        auto original = m.vertices;
        bool accepted = false;
        for (double alpha = 1; alpha >= 1e-12; alpha *= 0.5) {
            for (size_t i = 0; i < n; ++i)
                if (!m.boundary[i])
                    m.vertices[i] = original[i] + m.normal * (alpha * step[i]);
            double area = evaluate(m, false).area;
            if (area <= e.area + 1e-4 * alpha * descent) {
                accepted = true;
                break;
            }
        }
        if (!accepted) {
            m.vertices = std::move(original);
            if (observer)
                observer(m, iteration, result.areas.back(), residual, true);
            return result;
        }
    }
    return result;
}

Result minimize_spatial(Mesh &m, int max_iterations, double tolerance,
                        const IterationObserver &observer) {
    if (max_iterations < 1 || !(tolerance > 0) || !std::isfinite(tolerance))
        throw std::runtime_error("Некорректные параметры пространственной оптимизации.");
    const double minimum_area_floor = evaluate_spatial(m, false).minimum_twice_area * 1e-8;
    constexpr size_t memory = 8;
    std::vector<std::vector<double>> steps, gradient_changes;
    std::vector<double> inverse_curvatures, previous_x, previous_gradient;
    bool have_previous = false;
    Result result;
    auto coordinates = [&]() {
        std::vector<double> x(m.vertices.size() * 3);
        for (size_t i = 0; i < m.vertices.size(); ++i) {
            x[3 * i] = m.vertices[i].x;
            x[3 * i + 1] = m.vertices[i].y;
            x[3 * i + 2] = m.vertices[i].z;
        }
        return x;
    };
    for (int iteration = 0; iteration <= max_iterations; ++iteration) {
        auto e = evaluate_spatial(m);
        result.areas.push_back(e.area * m.scale * m.scale);
        result.iterations = iteration;
        std::vector<double> x = coordinates(), gradient(x.size());
        double residual = 0;
        for (size_t i = 0; i < m.vertices.size(); ++i) {
            Vec3 g = m.boundary[i] ? Vec3{} : e.gradient[i];
            gradient[3 * i] = g.x;
            gradient[3 * i + 1] = g.y;
            gradient[3 * i + 2] = g.z;
            residual = std::max(residual, norm(g));
        }
        result.residual = residual;
        if (observer)
            observer(m, iteration, result.areas.back(), residual, false);
        if (residual <= tolerance) {
            result.converged = true;
            if (observer)
                observer(m, iteration, result.areas.back(), residual, true);
            return result;
        }
        if (iteration == max_iterations) {
            if (observer)
                observer(m, iteration, result.areas.back(), residual, true);
            return result;
        }
        if (have_previous) {
            std::vector<double> s(x.size()), y(x.size());
            for (size_t i = 0; i < x.size(); ++i) {
                s[i] = x[i] - previous_x[i];
                y[i] = gradient[i] - previous_gradient[i];
            }
            double sy = inner(s, y), ss = inner(s, s), yy = inner(y, y);
            if (sy > 1e-14 * std::sqrt(ss * yy) && std::isfinite(sy)) {
                if (steps.size() == memory) {
                    steps.erase(steps.begin());
                    gradient_changes.erase(gradient_changes.begin());
                    inverse_curvatures.erase(inverse_curvatures.begin());
                }
                steps.push_back(std::move(s));
                gradient_changes.push_back(std::move(y));
                inverse_curvatures.push_back(1 / sy);
            }
        }
        std::vector<double> direction = gradient;
        std::vector<double> coefficients(steps.size());
        for (size_t k = steps.size(); k-- > 0;) {
            coefficients[k] = inverse_curvatures[k] * inner(steps[k], direction);
            for (size_t i = 0; i < direction.size(); ++i)
                direction[i] -= coefficients[k] * gradient_changes[k][i];
        }
        if (!steps.empty()) {
            double sy = 1 / inverse_curvatures.back();
            double yy = inner(gradient_changes.back(), gradient_changes.back());
            double scale = yy > 0 ? sy / yy : 1;
            for (double &value : direction)
                value *= scale;
        }
        for (size_t k = 0; k < steps.size(); ++k) {
            double beta = inverse_curvatures[k] * inner(gradient_changes[k], direction);
            for (size_t i = 0; i < direction.size(); ++i)
                direction[i] += steps[k][i] * (coefficients[k] - beta);
        }
        for (double &value : direction)
            value *= -1;
        double descent = inner(gradient, direction);
        if (!(descent < 0) || !std::isfinite(descent)) {
            direction = gradient;
            for (double &value : direction)
                value *= -1;
            descent = -inner(gradient, gradient);
            steps.clear();
            gradient_changes.clear();
            inverse_curvatures.clear();
        }
        auto original = m.vertices;
        double step_length = 1;
        bool accepted = false;
        for (int line_search = 0; line_search < 32; ++line_search) {
            for (size_t i = 0; i < m.vertices.size(); ++i) {
                if (m.boundary[i])
                    continue;
                m.vertices[i] = {original[i].x + step_length * direction[3 * i],
                                 original[i].y + step_length * direction[3 * i + 1],
                                 original[i].z + step_length * direction[3 * i + 2]};
            }
            try {
                auto candidate = evaluate_spatial(m, false);
                if (candidate.minimum_twice_area >= minimum_area_floor &&
                    candidate.area <= e.area + 1e-4 * step_length * descent) {
                    accepted = true;
                    break;
                }
            } catch (const std::runtime_error &) {
            }
            step_length *= 0.5;
        }
        if (!accepted) {
            m.vertices = std::move(original);
            if (observer)
                observer(m, iteration, result.areas.back(), residual, true);
            return result;
        }
        previous_x = std::move(x);
        previous_gradient = std::move(gradient);
        have_previous = true;
    }
    return result;
}
void write_obj(const Mesh &m, const std::string &path) {
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("Не удалось записать OBJ: " + path);
    out << std::setprecision(17)
        << "# Minimal-area piecewise linear graph; fixed polygonal boundary\n";
    for (auto p : m.vertices) {
        p = m.origin + p * m.scale;
        out << "v " << p.x << ' ' << p.y << ' ' << p.z << '\n';
    }
    for (auto f : m.faces)
        out << "f " << f[0] + 1 << ' ' << f[1] + 1 << ' ' << f[2] + 1 << '\n';
    if (!out)
        throw std::runtime_error("Ошибка записи OBJ.");
}
} // namespace minimal
