cmake_minimum_required (VERSION 2.6 FATAL_ERROR)

project(cluster-and-fit)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

find_package(Boost REQUIRED)
find_package (VTK REQUIRED)
find_package (PCL 1.8 REQUIRED)
find_package( OpenCV REQUIRED )


include_directories( ${OpenCV_INCLUDE_DIRS} )

#find_library(SIFTGPU_LIBRARY OpenIGTLink)
include_directories (include)

include_directories (${PCL_INCLUDE_DIRS})
link_directories    (${PCL_LIBRARY_DIRS})
add_definitions     (${PCL_DEFINITIONS})
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
find_package (Qt4 REQUIRED)
#set  (VTK_LIBRARIES   vtkRendering vtkGraphics vtkHybrid QVTK)
include(${VTK_USE_FILE})

set  (project_SOURCES main.cpp)
set  (project_HEADERS)
set  (project_FORMS)

ADD_EXECUTABLE  (cluster-fit  ${project_SOURCES}
        ${project_FORMS_HEADERS}
        ${project_HEADERS_MOC})
#ADD_EXECUTABLE( main2 src/file.cpp)

TARGET_LINK_LIBRARIES (cluster-fit ${PCL_LIBRARIES} ${Glue} ${VTK_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})




