cmake_minimum_required(VERSION 3.5)

project(MultilayerAbobatron VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

set(PROJECT_SOURCES
        main.cc
        Model/Dataloader.h
        Model/Dataloader.cc
        Model/MLPCore.h
        Model/MLPCore.cc
        Model/MatrixMLP/MLayer.h
        Model/MatrixMLP/MatrixMLP.h
        Model/MatrixMLP/MatrixMLP.cc
        Model/GraphMLP/Glayer.h
        Model/GraphMLP/Glayer.cc
        Model/GraphMLP/GraphMLP.h
        Model/GraphMLP/GraphMLP.cc
        Model/TrainingGround.h
        Model/TrainingGround.cc
        Model/TrainingConfig.h
        Model/MLPBuilder.h
        Model/MLPBuilder.cc
        View/mainwindow.cc
        View/mainwindow.h
        View/mainwindow.ui
        View/aboba.h
        View/aboba.cc
        View/aboba.ui
        View/TrainingSchedule.h
        View/TrainingSchedule.cc
        View/TrainingSchedule.ui
        View/DrawingWindow.h
        View/DrawingWindow.cc
        Controller/SettingsController.h
        Controller/MLPController.h
        Utility/ActivationFunction.h
        Utility/ActivationFunction.cc
        Utility/Matrix.h

)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(MultilayerAbobatron
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET MultilayerAbobatron APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(MultilayerAbobatron SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(MultilayerAbobatron
            ${PROJECT_SOURCES}
        )
    endif()
endif()
target_compile_options(MultilayerAbobatron PRIVATE
    -O3 -Wall -Werror -Wextra -std=c++17
    # Add more options as needed
)

target_link_libraries(MultilayerAbobatron PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)

set_target_properties(MultilayerAbobatron PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(MultilayerAbobatron)
endif()
