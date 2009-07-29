#
# Find the METIS includes and libraries
#
# METIS is an library that implements a variety of algorithms for
# partitioning unstructured graphs, meshes, and for computing 
# fill-reducing orderings of
# sparse matrices. It can be found at:
# 	http://www-users.cs.umn.edu/~karypis/metis/
#
# METIS_INCLUDE_DIR - where to find autopack.h
# METIS_LIBRARIES   - List of fully qualified libraries to link against.
# METIS_FOUND       - Do not attempt to use if "no" or undefined.

# derived from code by Christophe Prud'homme.

FIND_PATH(METIS_INCLUDE_DIR metis.h
  /usr/local/include
  /usr/include
  /usr/include/metis
  )

FIND_LIBRARY(METIS_LIBRARY metis
  /usr/local/lib
  /usr/lib
  )

IF(METIS_INCLUDE_DIR AND METIS_LIBRARY)
  SET( METIS_FOUND "YES" )
ELSE()
  IF(METIS_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "METIS not found")
  ENDIF()
ENDIF()
