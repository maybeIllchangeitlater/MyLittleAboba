cmake_minimum_required(VERSION 3.25)
project(MultilayerAbobatron)

set(CMAKE_CXX_STANDARD 17)

add_executable(MultilayerAbobatron main.cpp dataloader.cc MLP.cc)
