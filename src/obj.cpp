#include "surface.hpp"
#include <algorithm>
#include <cmath>

#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace minimal {
Mesh read_obj(const std::string &path, Vec3 normal) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Не удалось открыть OBJ: " + path);
    Mesh m;
    std::string line;
    while (std::getline(in, line)) {
        line = line.substr(0, line.find('#'));
        std::istringstream row(line);
        std::string tag;
        row >> tag;
        if (tag == "v") {
            Vec3 p;
            std::string extra;
            if (!(row >> p.x >> p.y >> p.z) || (row >> extra))
                throw std::runtime_error("OBJ: нужны трёхмерные вершины v x y z.");
            m.vertices.push_back(p);
        } else if (tag == "f") {
            std::array<int, 3> f;
            std::string token;
            for (int i = 0; i < 3; ++i) {
                if (!(row >> token))
                    throw std::runtime_error("OBJ: нужны треугольные грани.");
                token = token.substr(0, token.find('/'));
                size_t used = 0;
                int id = std::stoi(token, &used);
                if (used != token.size() || id == 0)
                    throw std::runtime_error("OBJ: неверный индекс.");
                f[i] = id > 0 ? id - 1 : int(m.vertices.size()) + id;
                if (f[i] < 0 || f[i] >= int(m.vertices.size()))
                    throw std::runtime_error(
                        "OBJ: индекс вне диапазона; вершины должны предшествовать граням.");
            }
            if (row >> token)
                throw std::runtime_error(
                    "OBJ: сначала разбейте многоугольные грани на треугольники.");
            m.faces.push_back(f);
        } else if (tag == "l" || tag == "p" || tag == "curv" || tag == "surf")
            throw std::runtime_error(
                "OBJ: поддерживается треугольная поверхность, не линии или сплайны.");
    }
    if (m.faces.empty() || m.faces.size() > 200000)
        throw std::runtime_error("OBJ: нужно 1–200000 треугольников.");
    using Edge = std::pair<int, int>;
    std::map<Edge, std::vector<Edge>> edges;
    std::vector<std::vector<int>> neighbors(m.vertices.size());
    std::set<std::array<int, 3>> unique;
    for (auto f : m.faces) {
        auto sorted = f;
        std::sort(sorted.begin(), sorted.end());
        if (sorted[0] == sorted[1] || sorted[1] == sorted[2] || !unique.insert(sorted).second)
            throw std::runtime_error("OBJ: повторная или вырожденная грань.");
        for (int i = 0; i < 3; ++i) {
            int a = f[i], b = f[(i + 1) % 3];
            edges[std::minmax(a, b)].push_back({a, b});
            neighbors[a].push_back(b);
            neighbors[b].push_back(a);
        }
    }
    std::map<int, int> next;
    std::set<int> incoming;
    for (auto const &item : edges) {
        auto const &uses = item.second;
        if (uses.size() == 1) {
            if (!next.emplace(uses[0].first, uses[0].second).second ||
                !incoming.insert(uses[0].second).second)
                throw std::runtime_error("OBJ: граница не является простой петлёй.");
        } else if (uses.size() != 2 || uses[0].first != uses[1].second)
            throw std::runtime_error(
                "OBJ: сетка немногообразна или ориентация граней несогласована.");
    }
    if (next.size() < 3)
        throw std::runtime_error(
            "OBJ: нужна открытая поверхность с одной границей, не замкнутая оболочка.");
    std::vector<int> boundary;
    int first = next.begin()->first, current = first;
    do {
        boundary.push_back(current);
        auto it = next.find(current);
        if (it == next.end())
            throw std::runtime_error("OBJ: разрыв границы.");
        current = it->second;
        if (boundary.size() > next.size())
            throw std::runtime_error("OBJ: некорректная граница.");
    } while (current != first);
    if (boundary.size() != next.size())
        throw std::runtime_error(
            "OBJ: несколько границ или отдельные компоненты не поддерживаются.");
    std::set<int> visited;
    std::vector<int> stack{first};
    while (!stack.empty()) {
        int v = stack.back();
        stack.pop_back();
        if (!visited.insert(v).second)
            continue;
        for (int w : neighbors[v])
            stack.push_back(w);
    }
    if (visited.size() != m.vertices.size() ||
        long(m.vertices.size()) - long(edges.size()) + long(m.faces.size()) != 1)
        throw std::runtime_error(
            "OBJ: нужна связная сетка топологии диска без неиспользуемых вершин.");
    std::vector<Vec3> contour;
    for (int i : boundary)
        contour.push_back(m.vertices[i]);
    auto frame = triangulate(contour, normal);
    m.origin = frame.origin;
    m.scale = frame.scale;
    m.normal = frame.normal;
    m.boundary.assign(m.vertices.size(), false);
    for (int i : boundary)
        m.boundary[i] = true;
    for (auto &p : m.vertices)
        p = (p - m.origin) * (1 / m.scale);
    double orientation = 0;
    for (auto f : m.faces) {
        double signed_area =
            dot(cross(m.vertices[f[1]] - m.vertices[f[0]], m.vertices[f[2]] - m.vertices[f[0]]),
                m.normal);
        if (!std::isfinite(signed_area) || std::abs(signed_area) < 1e-14)
            throw std::runtime_error("OBJ: вырожденная грань в проекции.");
        if (orientation == 0)
            orientation = signed_area;
        if (signed_area * orientation <= 0)
            throw std::runtime_error(
                "OBJ: поверхность с нависаниями или перевёрнутыми гранями не поддерживается.");
    }
    if (orientation < 0)
        for (auto &f : m.faces)
            std::swap(f[1], f[2]);
    return m;
}
} // namespace minimal
