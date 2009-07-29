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

SET(METIS_DIR "/usr" CACHE PATH "Root directory for the Metis install")

FIND_PATH(METIS_INCLUDE_DIR metis.h
  "${METIS_DIR}/include"
  /usr/local/include
  /usr/include
  /usr/include/metis
  )

FIND_LIBRARY(METIS_LIBRARY metis
  ${METIS_DIR}/lib/
  ${METIS_DIR}/build/Linux-i686/
  ${METIS_DIR}/build/Linux-x86_64/
  /usr/local/lib
  /usr/lib
  )

MESSAGE("${METIS_INCLUDE_DIR} yo")
MESSAGE("${METIS_LIBRARY} yo")

IF(METIS_INCLUDE_DIR AND METIS_LIBRARY)
  SET( METIS_FOUND "YES" )
ELSE()
  IF(METIS_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "METIS not found")
  ENDIF()
ENDIF()

MARK_AS_ADVANCED(METIS_INCLUDE_DIR METIS_LIBRARY)
