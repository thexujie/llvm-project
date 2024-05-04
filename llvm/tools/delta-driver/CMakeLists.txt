set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(delta-driver
  delta-driver.cpp

  DEPENDS
  )
