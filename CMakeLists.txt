cmake_minimum_required(VERSION 3.10)
project(Triangulation)

# Указываем стандарт C++
set(CMAKE_CXX_STANDARD 17)

# Указываем путь к SFML (замените на /usr/local для Mac с Intel)
set(SFML_DIR "/opt/homebrew/Cellar/sfml/2.5.1/lib/cmake/SFML")

# Находим пакеты SFML
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# Добавляем исходный файл
add_executable(triangulation main.cpp)

# Связываем библиотеки SFML с проектом
target_link_libraries(triangulation sfml-graphics sfml-window sfml-system)

# Устанавливаем пути к заголовочным файлам
target_include_directories(triangulation PRIVATE ${SFML_INCLUDE_DIR})
