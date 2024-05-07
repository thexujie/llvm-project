//===-- CharSet.h - Characters set conversion class ---------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides a utility class to convert between different character
/// set encodings.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CHARSET_H
#define LLVM_SUPPORT_CHARSET_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/config.h"
#include "llvm/Support/ErrorOr.h"

#include <functional>
#include <string>
#include <system_error>

namespace llvm {

template <typename T> class SmallVectorImpl;

namespace details {
class CharSetConverterImplBase {
public:
  virtual ~CharSetConverterImplBase() = default;

  /// Converts a string.
  /// \param[in] Source source string
  /// \param[out] Result container for converted string
  /// \return error code in case something went wrong
  ///
  /// The following error codes can occur, among others:
  ///   - std::errc::argument_list_too_long: The result requires more than
  ///     std::numeric_limits<size_t>::max() bytes.
  ///   - std::errc::illegal_byte_sequence: The input contains an invalid
  ///     multibyte sequence.
  ///   - std::errc::invalid_argument: The input contains an incomplete
  ///     multibyte sequence.
  ///
  /// If the destination charset is a stateful character set, the shift state
  /// will be set to the initial state.
  ///
  /// In case of an error, the result string contains the successfully converted
  /// part of the input string.
  ///

  virtual std::error_code convert(StringRef Source,
                                  SmallVectorImpl<char> &Result) const = 0;
};
} // namespace details

// Names inspired by https://wg21.link/p1885.
namespace text_encoding {
enum class id {
  /// UTF-8 character set encoding.
  UTF8,

  /// IBM EBCDIC 1047 character set encoding.
  IBM1047
};
} // end namespace text_encoding

/// Utility class to convert between different character set encodings.
class CharSetConverter {
  std::unique_ptr<details::CharSetConverterImplBase> Converter;

  CharSetConverter(std::unique_ptr<details::CharSetConverterImplBase> Converter)
      : Converter(std::move(Converter)) {}

public:
  /// Creates a CharSetConverter instance.
  /// \param[in] CSFrom the source character encoding
  /// \param[in] CSTo the target character encoding
  /// \return a CharSetConverter instance
  static CharSetConverter create(text_encoding::id CSFrom,
                                 text_encoding::id CSTo);

  /// Creates a CharSetConverter instance.
  /// Returns std::errc::invalid_argument in case the requested conversion is
  /// not supported.
  /// \param[in] CPFrom name of the source character encoding
  /// \param[in] CPTo name of the target character encoding
  /// \return a CharSetConverter instance or an error code
  static ErrorOr<CharSetConverter> create(StringRef CPFrom, StringRef CPTo);

  CharSetConverter(const CharSetConverter &) = delete;
  CharSetConverter &operator=(const CharSetConverter &) = delete;

  CharSetConverter(CharSetConverter &&Other)
      : Converter(std::move(Other.Converter)) {}

  CharSetConverter &operator=(CharSetConverter &&Other) {
    if (this != &Other)
      Converter = std::move(Other.Converter);
    return *this;
  }

  ~CharSetConverter() = default;

  /// Converts a string.
  /// \param[in] Source source string
  /// \param[out] Result container for converted string
  /// \return error code in case something went wrong
  std::error_code convert(StringRef Source,
                          SmallVectorImpl<char> &Result) const {
    return Converter->convert(Source, Result);
  }

  ErrorOr<std::string> convert(StringRef Source) const {
    SmallString<100> Result;
    auto EC = Converter->convert(Source, Result);
    if (!EC)
      return std::string(Result);
    return EC;
  }
};

} // namespace llvm

#endif
