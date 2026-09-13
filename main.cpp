#include "surface.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
    try {
        std::string contour, obj, prefix = "surface", mode = "graph";
        int levels = 3, iterations = 100;
        double tolerance = 1e-9;
        minimal::Vec3 normal;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto value = [&]() {
                if (++i == argc)
                    throw std::runtime_error("Не задано значение " + arg);
                return std::string(argv[i]);
            };
            auto integer = [&]() {
                auto s = value();
                size_t n = 0;
                int v = std::stoi(s, &n);
                if (n != s.size())
                    throw std::runtime_error("Ожидалось целое число.");
                return v;
            };
            auto real = [&]() {
                auto s = value();
                size_t n = 0;
                double v = std::stod(s, &n);
                if (n != s.size() || !std::isfinite(v))
                    throw std::runtime_error("Ожидалось конечное число.");
                return v;
            };
            if (arg == "--contour")
                contour = value();
            else if (arg == "--mesh")
                obj = value();
            else if (arg == "--output")
                prefix = value();
            else if (arg == "--mode")
                mode = value();
            else if (arg == "--refine")
                levels = integer();
            else if (arg == "--iterations")
                iterations = integer();
            else if (arg == "--tolerance")
                tolerance = real();
            else if (arg == "--normal") {
                normal.x = real();
                normal.y = real();
                normal.z = real();
                if (minimal::norm(normal) == 0)
                    throw std::runtime_error("Нулевая нормаль.");
            } else if (arg == "--help") {
                std::cout
                    << "Минимальная поверхность с закреплённой границей\n"
                       "triangulation [--contour points.csv | --mesh input.obj] [--normal nx ny "
                       "nz]\n"
                       "              [--mode graph|spatial]\n"
                       "              [--refine 0..6] [--iterations 100] [--tolerance 1e-9] "
                       "[--output surface]\n"
                       "Без входного файла: x=cos(t), y=sin(t), z=0.35*cos(2t).\n"
                       "Результаты: .obj, анимированный .html, .csv. "
                       "Формулы: python3 tools/formula.py --help\n";
                return 0;
            } else
                throw std::runtime_error("Неизвестный аргумент: " + arg);
        }
        if (!contour.empty() && !obj.empty())
            throw std::runtime_error("Выберите --contour или --mesh.");
        if (mode != "graph" && mode != "spatial")
            throw std::runtime_error("Режим должен быть graph или spatial.");
        if (levels < 0 || levels > 6 || iterations < 1 || iterations > 10000 || !(tolerance > 0))
            throw std::runtime_error("Недопустимые параметры расчёта.");
        minimal::Mesh mesh;
        if (!obj.empty())
            mesh = minimal::read_obj(obj, normal, mode == "graph");
        else {
            std::vector<minimal::Vec3> points;
            if (!contour.empty())
                points = minimal::read_contour(contour);
            else
                for (int i = 0; i < 48; ++i) {
                    double t = 2 * std::acos(-1.0) * i / 48;
                    points.push_back({std::cos(t), std::sin(t), 0.35 * std::cos(2 * t)});
                }
            mesh = minimal::triangulate(points, normal);
        }
        std::ofstream history(prefix + ".csv");
        if (!history)
            throw std::runtime_error("Не удалось открыть журнал результата.");
        history << "level,iteration,area\n" << std::setprecision(17);
        bool converged = false;
        std::vector<minimal::AnimationTopology> topologies;
        std::vector<minimal::AnimationFrame> frames;
        size_t stored_points = 0;
        constexpr size_t animation_point_budget = 250'000;
        for (int level = 0; level <= levels; ++level) {
            if (level)
                minimal::refine(mesh);
            const size_t topology = topologies.size();
            topologies.push_back({mesh.faces});
            auto record = [&](const minimal::Mesh &state, int iteration, double area,
                              double residual, bool final) {
                if (!frames.empty() && frames.back().topology == topology &&
                    frames.back().iteration == iteration) {
                    frames.back().vertices = state.vertices;
                    frames.back().area = area;
                    frames.back().residual = residual;
                    return;
                }
                const size_t points = state.vertices.size();
                if (stored_points + points > animation_point_budget && !final)
                    return;
                while (stored_points + points > animation_point_budget && frames.size() > 1) {
                    stored_points -= frames[1].vertices.size();
                    frames.erase(frames.begin() + 1);
                }
                if (stored_points + points > animation_point_budget)
                    return;
                frames.push_back({state.vertices, topology, area, residual, level, iteration});
                stored_points += points;
            };
            auto result = mode == "spatial"
                              ? minimal::minimize_spatial(mesh, iterations, tolerance, record)
                              : minimal::minimize(mesh, iterations, tolerance, record);
            converged = result.converged;
            for (size_t i = 0; i < result.areas.size(); ++i)
                history << level << ',' << i << ',' << result.areas[i] << '\n';
            std::cout << std::setprecision(12) << "Режим " << mode << ", уровень " << level
                      << ": вершин " << mesh.vertices.size() << ", треугольников "
                      << mesh.faces.size() << ", площадь " << result.areas.front() << " -> "
                      << result.areas.back() << ", невязка " << result.residual << ", итераций "
                      << result.iterations << ", " << (converged ? "сошлось" : "НЕ сошлось")
                      << '\n';
            if (!converged)
                break;
        }
        if (!history)
            throw std::runtime_error("Ошибка записи журнала.");
        minimal::write_obj(mesh, prefix + ".obj");
        minimal::write_html(mesh, prefix + ".html", topologies, frames);
        std::cout << "Сохранены " << prefix << ".obj, .html, .csv\n";
        return converged ? 0 : 2;
    } catch (std::exception const &e) {
        std::cerr << "Ошибка: " << e.what() << '\n';
        return 1;
    }
}
