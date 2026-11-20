set(MTS_PLUGIN_PREFIX "integrators")

# Permit lowering optimization for the MPG plugin alone on Windows smoke builds to dodge long single-TU compiles.
option(MTS_MPG_FAST_ITERATION "Enable faster local iteration flags for MPG on MSVC" OFF)

add_plugin(depth   depth.cpp)
add_plugin(direct  direct.cpp)
add_plugin(path    path.cpp)
add_plugin(ppg     ppg.cpp)
add_plugin(vapg    vapg.cpp)
add_plugin(vmmpg   vmmpg.cpp)
add_plugin(aov     aov.cpp)
add_plugin(stokes  stokes.cpp)
add_plugin(moment  moment.cpp)
add_plugin(volpath  volpath.cpp)
add_plugin(volpathmis volpathmis.cpp)


# Register the test directory
add_tests(${CMAKE_CURRENT_SOURCE_DIR}/tests)

# --- Manifold Path Guiding integrator (MPG) ---
add_plugin(manifold_path_guiding
  MPG/manifold_path_guiding.cpp
  MPG/manifold_path_guiding.h
  MPG/masked_diffusion_runtime.h
  MPG/ann.h
  MPG/chain_distribution.h
  MPG/dtree.h
  MPG/spatial_structure.h
  MPG/util.h
)

if (MTS_MPG_FAST_ITERATION AND MSVC)
  target_compile_options(manifold_path_guiding-obj PRIVATE /Od /Ob0 /utf-8)
endif()

if (MSVC)
  target_compile_options(ppg-obj PRIVATE /utf-8)
  target_compile_options(vapg-obj PRIVATE /utf-8)
  target_compile_options(vmmpg-obj PRIVATE /utf-8)
endif()
