# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")
load(":vulkan_sdk.bzl", "vulkan_sdk_setup")
load(":configure.bzl", "llvm_configure", "DEFAULT_TARGETS")

def _llvm_configure_extension_impl(ctx):
    targets = []
    patches = []
    commit = ""
    sha256 = ""

    for module in ctx.modules:

        # Aggregate targets and patches across imports.
        for config in module.tags.configure:
            for target in config.targets:
                if target not in targets:
                    targets.append(target)
            for patch in config.patches:
                if patch not in patches:
                    patches.append(patch)

        # Use the first nonempty commit/sha configuration, starting from the
        # top-level import and working down to the MODULE.bazel of the
        # llvm-project-overlay itself, which does not specify these values.
        # This way in-tree builds will always use the current sources and not
        # fetch the llvm repository from a hardcoded commit.
        if module.tags.configure != [] and commit == "":
            commit = module.tags.configure[0].commit
            sha256 = module.tags.configure[0].sha256

    if commit == "":
        if patches != []:
            fail("""Cannot apply patches when `commit` is unspecified. Patches
                 would modify some unknown version of the LLVM sources, making
                 the build irreproducible.
                 """,
            )
        new_local_repository(
            name = "llvm-raw",
            path = "../../",
            build_file_content = "#Empty.",
        )
    else:
        http_archive(
            name = "llvm-raw",
            build_file_content = "# Empty.",
            sha256 = sha256,
            strip_prefix = "llvm-project-" + commit,
            urls = [
                "https://github.com/llvm/llvm-project/archive/{}.tar.gz".format(
                    commit,
                ),
            ],
            patches = patches,
            patch_args = ["-p1"],
        )

    # Fall back to the default targets if all configurations of this extension
    # omit the `target` attribute.
    if targets == []:
        targets = DEFAULT_TARGETS

    llvm_configure(name = "llvm-project", targets = targets)

    http_archive(
        name = "llvm_zlib",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
        sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
        strip_prefix = "zlib-ng-2.0.7",
        urls = [
        "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
        ],
    )

    http_archive(
        name = "llvm_zstd",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
        sha256 = "9c4396cc829cfae319a6e2615202e82aad41372073482fce286fac78646d3ee4",
        strip_prefix = "zstd-1.5.5",
        urls = [
            "https://github.com/facebook/zstd/releases/download/v1.5.5/zstd-1.5.5.tar.gz"
        ],
    )

    http_archive(
        name = "vulkan_headers",
        build_file = "@llvm-raw//utils/bazel/third_party_build:vulkan_headers.BUILD",
        sha256 = "19f491784ef0bc73caff877d11c96a48b946b5a1c805079d9006e3fbaa5c1895",
        strip_prefix = "Vulkan-Headers-9bd3f561bcee3f01d22912de10bb07ce4e23d378",
        urls = [
            "https://github.com/KhronosGroup/Vulkan-Headers/archive/9bd3f561bcee3f01d22912de10bb07ce4e23d378.tar.gz",
        ],
    )

    vulkan_sdk_setup(name = "vulkan_sdk")

llvm_project_overlay = module_extension(
    doc = """Configure the llvm-project.

    Tags:
        targets: List of targets which Clang should support.
        commit: An optional LLVM commit. If specified, the LLVM project will be
            redownloaded from that commit (i.e.) it is downloaded once for the
            bzlmod dependency and once for the specified commit.
            Defaults to None, in which case the sources specified in the BCR
            module are used.
        sha256: Hash for verifying the custom archive if `commit` is not `None`.
        patches: Optional list of patches to apply to the LLVM sources. May not
            be used if `commit` is unspecified.
            """,
    implementation = _llvm_configure_extension_impl,
    tag_classes = {
        "configure": tag_class(
            attrs = {
                "commit": attr.string(),
                "sha256": attr.string(),
                "targets": attr.string_list(),
                "patches": attr.label_list(),
            },
        ),
    },
)
