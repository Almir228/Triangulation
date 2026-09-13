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
static double inner(const std::vector<double> &a, const std::vector<double> &b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}
Result minimize(Mesh &m, int max_iterations, double tolerance) {
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
        if (residual <= tolerance) {
            result.converged = true;
            return result;
        }
        if (iteration == max_iterations)
            return result;
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
            return result;
        }
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
