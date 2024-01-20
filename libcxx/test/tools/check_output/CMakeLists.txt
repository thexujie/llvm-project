
add_executable(check_output check_output.cpp)

# Link against libc++
if (TARGET cxx_shared)
  target_link_libraries(check_output PRIVATE cxx_shared)
elseif (TARGET cxx_static)
  target_link_libraries(check_output PRIVATE cxx_static)
else()
  message(FATAL "Neither cxx_shared nor cxx_static are available to be linked against")
endif()

# put the binary into <build>/bin
set_target_properties(check_output PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# set the language mode
set_target_properties(check_output PROPERTIES
                      CXX_STANDARD 23
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENSIONS NO)

target_compile_options(check_output PRIVATE -Werror=missing-prototypes)
