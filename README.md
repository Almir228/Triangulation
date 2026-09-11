# Triangulation: triangle geometry prototype

A small C++17 learning project exploring 3D vectors and triangle geometry. **The current executable improves the side-length balance of one hard-coded triangle; it does not construct a triangulation of a point cloud or a mesh.**

## Problem and mathematical idea

For triangle side lengths `a`, `b`, `c`, the code measures deviation from an equilateral shape as

```text
(max(a, b, c) - min(a, b, c)) / max(a, b, c)
```

`improveEquilateral` repeatedly moves one endpoint of the longest side a fraction `delta` toward the other endpoint. It stops when this deviation is below `tolerance`. Defaults are `delta = 1e-4` and `tolerance = 1e-3`. This is a heuristic that changes vertex positions and may shrink the triangle; it is not an optimisation proof or a mesh-quality guarantee.

## Structure and algorithm

- `Point3D`: vector arithmetic, distance and coordinate output.
- `Triangle`: area from a cross product, side lengths, shape score and iterative adjustment.
- `contains`: barycentric-coordinate calculation; coplanarity and degenerate inputs are not checked.
- `isDelaunaySatisfied`: experimental determinant helper, unused by the demo. Its orientation/3D behaviour is not validated and should not be treated as a correct general Delaunay predicate.
- `main.cpp`: all of the above plus the fixed example.
- `CMakeLists.txt`: builds one standard-library-only executable. The former SFML dependency was unused and has been removed.

## Build and run

Requires a C++17 compiler and CMake 3.10+:

```bash
cmake -S . -B build
cmake --build build
./build/triangulation
```

Or: `c++ -std=c++17 main.cpp -o triangulation` followed by `./triangulation`.

## Example input / output

Input is defined in `main()`, not read from stdin: `(0,0,6)`, `(1,0,0)`, `(0,3,0)`.

Observed output with the default parameters (last digits may vary):

```text
Before deviation from equilateral: 0.528595
Triangle points:
Point(0.56623, 0, 2.60262)
Point(1, 0, 0)
Point(0.0799803, 2.16928, 1.18155)
After deviation from equilateral: 0.000968854
```

## Relationship to Triangulation_1

[Triangulation_1](https://github.com/Almir228/Triangulation_1) is a later experimental variation: it adds vector helpers and a `Triangles` container, replaces `contains`, and removes the vertex-update step from `improveEquilateral`. Its demo leaves the shape unchanged. This repository is the recommended baseline because its demonstrated adjustment works. Commit messages do not establish the author's reason for creating two repositories.

## Limitations and next work

There is no point-set input, mesh export, visualisation, iteration limit or robust treatment of zero-length/collinear inputs. Invalid parameters may prevent convergence. Next steps are boundary-case tests and robust geometric predicates before implementing an actual triangulation algorithm. Neither version is a finished numerical library.
