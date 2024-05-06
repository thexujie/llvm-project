//===--- Managarm.h - Managarm ToolChain Implementations --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Managarm.h"
#include "Arch/ARM.h"
#include "Arch/Mips.h"
#include "Arch/PPC.h"
#include "Arch/RISCV.h"
#include "CommonArgs.h"
#include "clang/Config/config.h"
#include "clang/Driver/Distro.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "llvm/Option/ArgList.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Path.h"
#include <system_error>

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

using tools::addPathIfExists;

/// \brief Get our best guess at the multiarch triple for a target.
std::string Managarm::getMultiarchTriple(const Driver &D,
                                         const llvm::Triple &TargetTriple,
                                         StringRef SysRoot) const {
  // For most architectures, just use whatever we have rather than trying to be
  // clever.
  switch (TargetTriple.getArch()) {
  default:
    break;

  case llvm::Triple::x86_64:
    // We don't want this for x32, otherwise it will match x86_64 libs
    return "x86_64-managarm";
  case llvm::Triple::aarch64:
    return "aarch64-managarm";
  case llvm::Triple::riscv64:
    return "riscv64-managarm";
  }
  return TargetTriple.str();
}

static StringRef getOSLibDir(const llvm::Triple &Triple, const ArgList &Args) {
  // It happens that only x86, PPC and SPARC use the 'lib32' variant of
  // oslibdir, and using that variant while targeting other architectures causes
  // problems because the libraries are laid out in shared system roots that
  // can't cope with a 'lib32' library search path being considered. So we only
  // enable them when we know we may need it.
  //
  // FIXME: This is a bit of a hack. We should really unify this code for
  // reasoning about oslibdir spellings with the lib dir spellings in the
  // GCCInstallationDetector, but that is a more significant refactoring.
  if (Triple.getArch() == llvm::Triple::x86 || Triple.isPPC32() ||
      Triple.getArch() == llvm::Triple::sparc)
    return "lib32";

  if (Triple.getArch() == llvm::Triple::x86_64 && Triple.isX32())
    return "libx32";

  if (Triple.getArch() == llvm::Triple::riscv32)
    return "lib32";

  return Triple.isArch32Bit() ? "lib" : "lib64";
}

Managarm::Managarm(const Driver &D, const llvm::Triple &Triple,
                   const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  GCCInstallation.init(Triple, Args);
  Multilibs = GCCInstallation.getMultilibs();
  SelectedMultilibs.assign({GCCInstallation.getMultilib()});
  std::string SysRoot = computeSysRoot();

  ToolChain::path_list &PPaths = getProgramPaths();
  Generic_GCC::PushPPaths(PPaths);

#ifdef ENABLE_LINKER_BUILD_ID
  ExtraOpts.push_back("--build-id");
#endif

  // The selection of paths to try here is designed to match the patterns which
  // the GCC driver itself uses, as this is part of the GCC-compatible driver.
  // This was determined by running GCC in a fake filesystem, creating all
  // possible permutations of these directories, and seeing which ones it added
  // to the link paths.
  path_list &Paths = getFilePaths();

  const std::string OSLibDir = std::string(getOSLibDir(Triple, Args));
  const std::string MultiarchTriple = getMultiarchTriple(D, Triple, SysRoot);

  Generic_GCC::AddMultilibPaths(D, SysRoot, OSLibDir, MultiarchTriple, Paths);

  addPathIfExists(D, concat(SysRoot, "/lib", MultiarchTriple), Paths);
  addPathIfExists(D, concat(SysRoot, "/lib/..", OSLibDir), Paths);
  addPathIfExists(D, concat(SysRoot, "/usr/lib/", MultiarchTriple), Paths);
  addPathIfExists(D, concat(SysRoot, "/usr/lib/../", OSLibDir), Paths);

  Generic_GCC::AddMultiarchPaths(D, SysRoot, OSLibDir, Paths);

  addPathIfExists(D, SysRoot + "/lib", Paths);
  addPathIfExists(D, SysRoot + "/usr/lib", Paths);
}

bool Managarm::HasNativeLLVMSupport() const { return true; }

Tool *Managarm::buildLinker() const {
  return new tools::gnutools::Linker(*this);
}

Tool *Managarm::buildAssembler() const {
  return new tools::gnutools::Assembler(*this);
}

std::string Managarm::computeSysRoot() const {
  if (!getDriver().SysRoot.empty())
    return getDriver().SysRoot;
  return std::string();
}

std::string Managarm::getDynamicLinker(const ArgList &Args) const {
  switch (getTriple().getArch()) {
  case llvm::Triple::aarch64:
    return "/lib/aarch64-managarm/ld.so";
  case llvm::Triple::riscv64: {
    StringRef ABIName = tools::riscv::getRISCVABI(Args, getTriple());
    return ("/lib/riscv64-managarm/ld-riscv64-" + ABIName + ".so").str();
  }
  case llvm::Triple::x86_64:
    return "/lib/x86_64-managarm/ld.so";
  default:
    llvm_unreachable("unsupported architecture");
  }
}

void Managarm::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                         ArgStringList &CC1Args) const {
  const Driver &D = getDriver();
  std::string SysRoot = computeSysRoot();

  if (DriverArgs.hasArg(clang::driver::options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  addSystemInclude(DriverArgs, CC1Args, SysRoot + "/usr/local/include");

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> P(D.ResourceDir);
    llvm::sys::path::append(P, "include");
    addSystemInclude(DriverArgs, CC1Args, P);
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  // LOCAL_INCLUDE_DIR
  addSystemInclude(DriverArgs, CC1Args, concat(SysRoot, "/usr/local/include"));
  // TOOL_INCLUDE_DIR
  AddMultilibIncludeArgs(DriverArgs, CC1Args);

  // Check for configure-time C include directories.
  StringRef CIncludeDirs(C_INCLUDE_DIRS);
  if (CIncludeDirs != "") {
    SmallVector<StringRef, 5> dirs;
    CIncludeDirs.split(dirs, ":");
    for (StringRef dir : dirs) {
      StringRef Prefix =
          llvm::sys::path::is_absolute(dir) ? StringRef(SysRoot) : "";
      addExternCSystemInclude(DriverArgs, CC1Args, Prefix + dir);
    }
    return;
  }

  // On systems using multiarch and Android, add /usr/include/$triple before
  // /usr/include.
  std::string MultiarchIncludeDir = getMultiarchTriple(D, getTriple(), SysRoot);
  if (!MultiarchIncludeDir.empty() &&
      D.getVFS().exists(concat(SysRoot, "/usr/include", MultiarchIncludeDir)))
    addExternCSystemInclude(
        DriverArgs, CC1Args,
        concat(SysRoot, "/usr/include", MultiarchIncludeDir));

  // Add an include of '/include' directly. This isn't provided by default by
  // system GCCs, but is often used with cross-compiling GCCs, and harmless to
  // add even when Clang is acting as-if it were a system compiler.
  addExternCSystemInclude(DriverArgs, CC1Args, concat(SysRoot, "/include"));

  addExternCSystemInclude(DriverArgs, CC1Args, concat(SysRoot, "/usr/include"));
}

void Managarm::addLibStdCxxIncludePaths(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args) const {
  // We need a detected GCC installation on Managarm to provide libstdc++'s
  // headers.
  if (!GCCInstallation.isValid())
    return;

  StringRef TripleStr = GCCInstallation.getTriple().str();

  // Try generic GCC detection.
  Generic_GCC::addGCCLibStdCxxIncludePaths(DriverArgs, CC1Args, TripleStr);
}

SanitizerMask Managarm::getSupportedSanitizers() const {
  const bool IsX86 = getTriple().getArch() == llvm::Triple::x86;
  const bool IsX86_64 = getTriple().getArch() == llvm::Triple::x86_64;
  SanitizerMask Res = ToolChain::getSupportedSanitizers();
  Res |= SanitizerKind::PointerCompare;
  Res |= SanitizerKind::PointerSubtract;
  Res |= SanitizerKind::KernelAddress;
  Res |= SanitizerKind::Vptr;
  if (IsX86_64)
    Res |= SanitizerKind::KernelMemory;
  if (IsX86 || IsX86_64)
    Res |= SanitizerKind::Function;
  return Res;
}
