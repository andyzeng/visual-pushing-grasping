cmake_minimum_required(VERSION 2.8.9)
project (realsense)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_BUILD_TYPE "Release" )
set(CMAKE_BUILD_TYPE "Debug" )

# This parameter is meant for disabling graphical examples when building for
# save-to-disk targets.
option(BUILD_GRAPHICAL_EXAMPLES "Build graphical examples and tools." ON)

if(BUILD_GRAPHICAL_EXAMPLES)
    find_package(OpenGL)
    if(NOT OPENGL_FOUND)
        message(FATAL_ERROR "\n\n OpenGL package is missing!\n\n")
    endif()

    set(DEPENDENCIES realsense2 ${OPENGL_LIBRARIES})

    if(WIN32)
        list(APPEND DEPENDENCIES glfw3)
    else()
        # Find glfw header
        find_path(GLFW_INCLUDE_DIR NAMES GLFW/glfw3.h
            PATHS /usr/X11R6/include
                  /usr/include/X11
                  /opt/graphics/OpenGL/include
                  /opt/graphics/OpenGL/contrib/libglfw
                  /usr/local/include
                  /usr/include/GL
                  /usr/include
        )
        # Find glfw library
        find_library(GLFW_LIBRARIES NAMES glfw glfw3
                PATHS /usr/lib64
                      /usr/lib
                      /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
                      /usr/local/lib64
                      /usr/local/lib
                      /usr/local/lib/${CMAKE_LIBRARY_ARCHITECTURE}
                      /usr/X11R6/lib
        )
        list(APPEND DEPENDENCIES m ${GLFW_LIBRARIES} ${LIBUSB1_LIBRARIES})
        include_directories(${GLFW_INCLUDE_DIR})
    endif()
else()
    set(DEPENDENCIES realsense2)
    if(NOT WIN32)
        list(APPEND DEPENDENCIES m ${LIBUSB1_LIBRARIES})
    endif()
endif()


set(PROJECT_LINK_LIBS librealsense2.so)
add_executable(realsense realsense.cpp)
target_link_libraries(realsense pthread ${DEPENDENCIES} ${PROJECT_LINK_LIBS})
