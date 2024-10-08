#ifndef _NDARRAY_CONFIG_HH
#define _NDARRAY_CONFIG_HH

#define NDARRAY_VERSION "${NDARRAY_VERSION}"

//#cmakedefine NDARRAY_HAVE_ADIOS2 1
//#cmakedefine NDARRAY_HAVE_CUDA 1
//#cmakedefine NDARRAY_HAVE_HDF5 1
//#cmakedefine NDARRAY_HAVE_MPI 1
//#cmakedefine NDARRAY_HAVE_NETCDF 1
//#cmakedefine NDARRAY_HAVE_OPENMP 1
//#cmakedefine NDARRAY_HAVE_PNETCDF 1
//#cmakedefine NDARRAY_HAVE_PNG 1
//#cmakedefine NDARRAY_HAVE_VTK 1
//
//#cmakedefine NDARRAY_USE_BIG_ENDIAN 1
//#cmakedefine NDARRAY_USE_LITTLE_ENDIAN 1
#define NDARRAY_HAVE_NETCDF 1
#define NDARRAY_HAVE_VTK 1
#define NOMINMAX
#include <limits>
#include <algorithm>
#if NDARRAY_HAVE_MPI
#else
  typedef int MPI_Comm;
#define MPI_COMM_WORLD 0x44000000
#define MPI_COMM_SELF 0x44000001
#endif

#ifdef __CUDACC__
// #define NDARRAY_NUMERIC_FUNC __device__ __host__
#else
// #define NDARRAY_NUMERIC_FUNC
#define __device__ 
#define __host__ 
#endif

// utilities
#define NC_SAFE_CALL(call) {\
  int retval = call;\
  if (retval != 0) {\
    fprintf(stderr, "[NetCDF Error] %s, in file '%s', line %i.\n", nc_strerror(retval), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  }\
}

#define PNC_SAFE_CALL(call) {\
  int retval = call;\
  if (retval != 0) {\
      fprintf(stderr, "[PNetCDF Error] %s, in file '%s', line %i.\n", ncmpi_strerror(retval), __FILE__, __LINE__); \
      exit(EXIT_FAILURE); \
  }\
}

#endif
