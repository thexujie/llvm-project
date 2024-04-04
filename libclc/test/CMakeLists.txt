set(LIBCLC_TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

list(APPEND LIBCLC_TEST_DEPS
  libclc::clang
  FileCheck count not
)

add_custom_target(check-libclc)

foreach( t ${LIBCLC_TARGETS_TO_BUILD} )
  foreach( d ${${t}_devices} )

    # The tests use LLVM IR, so we can't test SPIR-V libraries here.
    string( REPLACE "-" ";" TRIPLE  ${t} )
    list( GET TRIPLE 0 ARCH )
    if( ARCH STREQUAL spirv OR ARCH STREQUAL spirv64 )
      continue()
    endif()

    get_libclc_device_info(
      TRIPLE ${t}
      DEVICE ${d}
      CPU cpu
      ARCH_SUFFIX arch_suffix
      CLANG_TRIPLE clang_triple
    )

    add_lit_testsuite(check-libclc-${arch_suffix}
      "Running libclc ${arch_suffix} regression tests"
      ${CMAKE_CURRENT_BINARY_DIR}
      PARAMS "target=${clang_triple}" cpu=${cpu}
        "builtins=$<TARGET_PROPERTY:prepare-${arch_suffix},TARGET_FILE>"
      DEPENDS prepare-${arch_suffix} ${LIBCLC_TEST_DEPS}
    )

    add_dependencies( check-libclc check-libclc-${arch_suffix} )

  endforeach()
endforeach()
