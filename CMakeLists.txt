cmake_minimum_required(VERSION 3.10)
project(Triangulation LANGUAGES CXX)

add_executable(triangulation main.cpp)
target_compile_features(triangulation PRIVATE cxx_std_17)
set_target_properties(triangulation PROPERTIES CXX_EXTENSIONS OFF)
