set(NSAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(NSAN_LIT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(NSAN_TESTSUITES)

set(NSAN_UNITTEST_DEPS)
set(NSAN_TEST_DEPS
  ${SANITIZER_COMMON_LIT_TEST_DEPS}
  nsan)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  )

foreach(arch ${NSAN_SUPPORTED_ARCH})
  set(NSAN_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}" NSAN_TEST_CONFIG_SUFFIX)
  get_test_cc_for_arch(${arch} NSAN_TEST_TARGET_CC NSAN_TEST_TARGET_CFLAGS)
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND NSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-nsan "Running the NSan tests"
  ${NSAN_TESTSUITES}
  DEPENDS ${NSAN_TEST_DEPS})
set_target_properties(check-nsan PROPERTIES FOLDER "Compiler-RT Misc")

