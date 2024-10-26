#include <iostream>
#include <cmath>
#include <vector>

class Point3D {
public:
    double x, y, z;

    // Конструктор
    Point3D(double x, double y, double z) : x(x), y(y), z(z) {}

    Point3D operator+(const Point3D& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    const Point3D operator-(const Point3D& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }
    const Point3D operator*(const double& a) const{
        return {a*x, a*y, a*z};
    }


    Point3D vector_tweak(Point3D vector, double delta , bool inplace = true) {
        if (inplace) {
            x+=delta*vector.x;
            y+=delta*vector.y;
            z+=delta*vector.z;
            return *this;
        } else {
            // Создаем новую точку с измененными координатами
            return {x + delta*vector.x, y + delta*vector.y, z + delta*vector.z};
        }
    }
    // Метод для вывода координат
    void print() const {
        std::cout << "Point(" << x << ", " << y << ", " << z << ")" << std::endl;
    }

    // Метод для вычисления расстояния до другой точки
    double distanceTo(const Point3D& other) const {
        return std::sqrt(std::pow(x - other.x, 2) + std::pow(y - other.y, 2) + std::pow(z - other.z, 2));
    }
};

class Triangle {
public:
    Point3D p1;
    Point3D p2;
    Point3D p3;
    // Конструктор принимает ссылки на объекты Point3D
    Triangle(const Point3D& a, const Point3D& b, const Point3D& c) : p1(a), p2(b), p3(c) {}


    bool contains(const Point3D& p) const {
        // Векторы от вершин треугольника к точке p
        Point3D v0 = p2 - p1;
        Point3D v1 = p3 - p1;
        Point3D v2 = p - p1;

        // Вычисляем скалярные произведения
        double dot00 = v0.x * v0.x + v0.y * v0.y + v0.z * v0.z;
        double dot01 = v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
        double dot02 = v0.x * v2.x + v0.y * v2.y + v0.z * v2.z;
        double dot11 = v1.x * v1.x + v1.y * v1.y + v1.z * v1.z;
        double dot12 = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;

        // Вычисляем барицентрические координаты
        double denom = dot00 * dot11 - dot01 * dot01;
        double u = (dot11 * dot02 - dot01 * dot12) / denom;
        double v = (dot00 * dot12 - dot01 * dot02) / denom;

        // Проверяем, находится ли точка внутри треугольника
        return (u >= 0) && (v >= 0) && (u + v <= 1);
    }
    
    bool isDelaunaySatisfied(const Point3D& p) const {
        // Координаты вершин треугольника и точки p
        double ax = p1.x - p.x;
        double ay = p1.y - p.y;
        double az = p1.z - p.z;

        double bx = p2.x - p.x;
        double by = p2.y - p.y;
        double bz = p2.z - p.z;

        double cx = p3.x - p.x;
        double cy = p3.y - p.y;
        double cz = p3.z - p.z;

        // Вычисляем матрицу для проверки условия Делоне
        double det = (ax * ax + ay * ay + az * az) * (bx * cy - by * cx)
                     - (bx * bx + by * by + bz * bz) * (ax * cy - ay * cx)
                     + (cx * cx + cy * cy + cz * cz) * (ax * by - ay * bx);

        // Условие Делоне выполнено, если определитель положителен
        return det > 0;
    }


    // Вычисление площади треугольника
    double area() const {
        // Вычисление площади через векторное произведение
        double ax = p2.x - p1.x;
        double ay = p2.y - p1.y;
        double az = p2.z - p1.z;

        double bx = p3.x - p1.x;
        double by = p3.y - p1.y;
        double bz = p3.z - p1.z;

        double cross_x = ay * bz - az * by;
        double cross_y = az * bx - ax * bz;
        double cross_z = ax * by - ay * bx;

        return 0.5 * std::sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
    }

    double sideLength(const Point3D& a, const Point3D& b) const {
        return a.distanceTo(b);
    }

    double isEquilateral() const {
        double side1 = sideLength(p1, p2);
        double side2 = sideLength(p2, p3);
        double side3 = sideLength(p3, p1);

        // Находим максимальную и минимальную длину
        double maxSide = std::max({side1, side2, side3});
        double minSide = std::min({side1, side2, side3});

        // Возвращаем относительное отклонение
        return (maxSide - minSide) / maxSide; // Чем меньше значение, тем более равносторонний треугольник
    }

    auto improveEquilateral(double delta = 1e-4, double tolerance = 1e-3, bool inplace = true) {
        Triangle improvedTriangle = *this; // Создаем копию текущего треугольника
        double deviation = improvedTriangle.isEquilateral(); // Текущее отклонение

        // Пока отклонение больше допустимого, продолжаем корректировать вершины
        while (deviation > tolerance) {
            double side1 = improvedTriangle.sideLength(improvedTriangle.p1, improvedTriangle.p2);
            double side2 = improvedTriangle.sideLength(improvedTriangle.p2, improvedTriangle.p3);
            double side3 = improvedTriangle.sideLength(improvedTriangle.p3, improvedTriangle.p1);

            double maxSide = std::max({side1, side2, side3});

            // Корректируем ту вершину, которая образует максимальную сторону
            if (side1 == maxSide) {
                improvedTriangle.p1 = improvedTriangle.p1 + (improvedTriangle.p2 - improvedTriangle.p1) * delta;
            } else if (side2 == maxSide) {
                improvedTriangle.p2 = improvedTriangle.p2 + (improvedTriangle.p3 - improvedTriangle.p2) * delta;
            } else if (side3 == maxSide) {
                improvedTriangle.p3 = improvedTriangle.p3 + (improvedTriangle.p1 - improvedTriangle.p3) * delta;
            }

            // Обновляем отклонение после корректировки
            deviation = improvedTriangle.isEquilateral();
        }

        if (inplace) {
            // Если inplace == true, изменяем текущий объект
            *this = improvedTriangle;
            return *this;
        } else {
            // Иначе возвращаем улучшенный треугольник
            return improvedTriangle;
        }
    }


    void print() const {
        std::cout << "Triangle points:\n";
        p1.print();
        p2.print();
        p3.print();
    }
};



int main() {
    Point3D a(0, 0, 6);
    Point3D b(1, 0, 0);
    Point3D c(0, 3, 0); // Равносторонний треугольник


    Triangle triangle(a, b, c); // Создаем треугольник

    double deviation = triangle.isEquilateral();
    std::cout << "Before deviation from equilateral: " << deviation << std::endl;

    triangle.improveEquilateral().print();
    deviation = triangle.isEquilateral();
    std::cout << "After deviation from equilateral: " << deviation << std::endl;
    return 0;
}
