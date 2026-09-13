cmake_minimum_required(VERSION 3.10)
project(Triangulation LANGUAGES CXX)

add_library(minimal_surface src/surface.cpp src/obj.cpp src/remesh.cpp src/viewer.cpp)
target_include_directories(minimal_surface PUBLIC src)
target_compile_features(minimal_surface PUBLIC cxx_std_17)
set_target_properties(minimal_surface PROPERTIES CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(minimal_surface PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(triangulation main.cpp)
target_link_libraries(triangulation PRIVATE minimal_surface)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(triangulation PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(surface_tests tests/surface_tests.cpp)
    target_link_libraries(surface_tests PRIVATE minimal_surface)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(surface_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME surface_numerics COMMAND surface_tests)

    find_package(Python3 3.9 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME formula_parser
            COMMAND ${Python3_EXECUTABLE} -m unittest discover
                    -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -p test_formula.py
        )
        add_test(
            NAME command_line
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cli.py
                    $<TARGET_FILE:triangulation>
        )
    endif()
endif()
