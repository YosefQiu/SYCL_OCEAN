cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(MPASO_SYCL VERSION 0.1.0)

set(PRJ_COMPILE_FEATURES)
set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)


FIND_PACKAGE(VTK COMPONENTS
  CommonColor
  CommonCore
  CommonDataModel
  FiltersCore
  FiltersSources
  FiltersGeometry
  IOLegacy
  IOXML
  InteractionStyle
  RenderingAnnotation
  RenderingContextOpenGL2
  RenderingCore
  RenderingFreeType
  RenderingGL2PSOpenGL2
  RenderingOpenGL2 QUIET
)
IF(NOT VTK_FOUND)
  MESSAGE(STATUS "VTK not found. Configuring VTK from the External/VTK directory.")
ENDIF()

find_package(yaml-cpp REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsycl -fno-fast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fno-fast-math")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsycl")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")

message(STATUS "CMAKE_C_COMPILER = ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS =  ${CMAKE_CXX_FLAGS}")
# Add Include Path
include_directories(
  third_part
  src
  include
  /usr/include  # Add the NetCDF include directory
)

# find source codes
# Add source files
file(GLOB_RECURSE SOURCE_FILES 
    src/*.c
    src/*.cpp
    src/*.cc)
if(WIN32)
    add_definitions(-D_AMD64_)
endif()
message("Following files are complied " ${SOURCE_FILES})

# Add NetCDF libraries
set(NetCDF_LIBRARIES "/usr/lib/x86_64-linux-gnu/libnetcdf.so" "/usr/lib/x86_64-linux-gnu/libnetcdf_c++4.so")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Link the NetCDF libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${NetCDF_LIBRARIES} ${VTK_LIBRARIES} yaml-cpp)
vtk_module_autoinit(
    TARGETS ${PROJECT_NAME}
    MODULES ${VTK_LIBRARIES}
)
target_compile_features(${PROJECT_NAME} PRIVATE ${PRJ_COMPILE_FEATURES})