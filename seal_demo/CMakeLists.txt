cmake_minimum_required(VERSION 3.10)
project(SEALDemo VERSION 1.0)
add_executable(sealdemo sealdemo.cpp)
add_executable(cubdemo seal_cub.cpp)

find_package(SEAL 4.1 REQUIRED)
target_link_libraries(sealdemo SEAL::seal)
target_link_libraries(cubdemo SEAL::seal)
