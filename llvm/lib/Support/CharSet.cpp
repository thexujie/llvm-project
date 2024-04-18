//===-- CharSet.cpp - Characters sets conversion class ------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides utility classes to convert between different character
/// set encoding.
///
//===----------------------------------------------------------------------===//

// UNICODE LICENSE V3
//
// COPYRIGHT AND PERMISSION NOTICE
//
// Copyright © 2016-2024 Unicode, Inc.
//
// NOTICE TO USER: Carefully read the following legal agreement. BY
// DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
// SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
// TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
// DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of data files and any associated documentation (the "Data Files") or
// software and any associated documentation (the "Software") to deal in the
// Data Files or Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, and/or sell
// copies of the Data Files or Software, and to permit persons to whom the
// Data Files or Software are furnished to do so, provided that either (a)
// this copyright and permission notice appear with all copies of the Data
// Files or Software, or (b) this copyright and permission notice appear in
// associated Documentation.
//
// THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
// KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
// THIRD PARTY RIGHTS.
//
// IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
// BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
// OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
// WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
// ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
// FILES OR SOFTWARE.

// Except as contained in this notice, the name of a copyright holder shall
// not be used in advertising or otherwise to promote the sale, use or other
// dealings in these Data Files or Software without prior written
// authorization of the copyright holder.

#include "llvm/Support/CharSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ConvertEBCDIC.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
#include <system_error>

#ifdef HAVE_ICU
#include <unicode/ucnv.h>
#elif defined(HAVE_ICONV)
#include <iconv.h>
#endif

using namespace llvm;

// Normalize the charset name with the charset alias matching algorithm proposed
// in https://www.unicode.org/reports/tr22/tr22-8.html#Charset_Alias_Matching.
static void normalizeCharSetName(StringRef CSName,
                                 SmallVectorImpl<char> &Normalized) {
  bool PrevDigit = false;
  for (auto Ch : CSName) {
    if (isAlnum(Ch)) {
      Ch = toLower(Ch);
      if (Ch != '0' || PrevDigit) {
        PrevDigit = isDigit(Ch);
        Normalized.push_back(Ch);
      }
    }
  }
}

// Maps the charset name to enum constant if possible.
std::optional<text_encoding::id> getKnownCharSet(StringRef CSName) {
  SmallString<16> Normalized;
  normalizeCharSetName(CSName, Normalized);
  if (Normalized.equals("utf8"))
    return text_encoding::id::UTF8;
  if (Normalized.equals("ibm1047"))
    return text_encoding::id::IBM1047;
  return std::nullopt;
}

void HandleOverflow(size_t &Capacity, char *&Output, size_t &OutputLength,
                    SmallVectorImpl<char> &Result) {
  // No space left in output buffer. Double the size of the underlying
  // memory in the SmallVectorImpl, adjust pointer and length and continue
  // the conversion.
  Capacity = (Capacity < std::numeric_limits<size_t>::max() / 2)
                 ? 2 * Capacity
                 : std::numeric_limits<size_t>::max();
  Result.resize_for_overwrite(Capacity);
  Output = static_cast<char *>(Result.data());
  OutputLength = Capacity;
}

namespace {
enum ConversionType {
  UTFToIBM1047,
  IBM1047ToUTF,
};

// Support conversion between EBCDIC 1047 and UTF8. This class uses
// built-in translation tables that allow for translation between the
// aforementioned character sets. The use of tables for conversion is only
// possible because EBCDIC 1047 is a single-byte, stateless encoding; other
// character sets are not supported.
class CharSetConverterTable : public details::CharSetConverterImplBase {
  ConversionType ConvType;

public:
  CharSetConverterTable(ConversionType ConvType) : ConvType(ConvType) {}

  std::error_code convert(StringRef Source, SmallVectorImpl<char> &Result,
                          bool ShouldAutoFlush) const override;
  std::error_code flush() const override;
  std::error_code flush(SmallVectorImpl<char> &Result) const override;
};

std::error_code CharSetConverterTable::convert(StringRef Source,
                                               SmallVectorImpl<char> &Result,
                                               bool ShouldAutoFlush) const {
  if (ConvType == IBM1047ToUTF) {
    ConverterEBCDIC::convertToUTF8(Source, Result);
    return std::error_code();
  } else if (ConvType == UTFToIBM1047) {
    return ConverterEBCDIC::convertToEBCDIC(Source, Result);
  }
  llvm_unreachable("Invalid ConvType!");
  return std::error_code();
}

std::error_code CharSetConverterTable::flush() const {
  return std::error_code();
}

std::error_code
CharSetConverterTable::flush(SmallVectorImpl<char> &Result) const {
  return std::error_code();
}

#ifdef HAVE_ICU
class CharSetConverterICU : public details::CharSetConverterImplBase {
  UConverter *FromConvDesc;
  UConverter *ToConvDesc;

public:
  CharSetConverterICU(UConverter *Converter) {
    UErrorCode EC = U_ZERO_ERROR;
    FromConvDesc = nullptr;
    ToConvDesc = ucnv_safeClone(Converter, nullptr, nullptr, &EC);
    if (U_FAILURE(EC)) {
      ToConvDesc = nullptr;
    }
  };

  CharSetConverterICU(UConverter *FromConverter, UConverter *ToConverter) {
    UErrorCode EC = U_ZERO_ERROR;
    FromConvDesc = ucnv_safeClone(FromConverter, nullptr, nullptr, &EC);
    if (U_FAILURE(EC))
      FromConvDesc = nullptr;
    ToConvDesc = ucnv_safeClone(ToConverter, nullptr, nullptr, &EC);
    if (U_FAILURE(EC))
      ToConvDesc = nullptr;
  }

  std::error_code convert(StringRef Source, SmallVectorImpl<char> &Result,
                          bool ShouldAutoFlush) const override;
  std::error_code flush() const override;
  std::error_code flush(SmallVectorImpl<char> &Result) const override;
};

std::error_code CharSetConverterICU::convert(StringRef Source,
                                             SmallVectorImpl<char> &Result,
                                             bool ShouldAutoFlush) const {
  // Setup the output. We directly write into the SmallVector.
  Result.resize_for_overwrite(Source.size());
  size_t OutputLength, Capacity = Result.capacity();
  char *Output, *Out;

  UErrorCode EC = U_ZERO_ERROR;

  do {
    EC = U_ZERO_ERROR;
    size_t InputLength = Source.size();
    const char *Input =
        InputLength ? const_cast<char *>(Source.data()) : nullptr;
    const char *In = Input;
    Output = InputLength ? static_cast<char *>(Result.data()) : nullptr;
    OutputLength = Capacity;
    Out = Output;
    Result.resize_for_overwrite(Capacity);
    ucnv_convertEx(ToConvDesc, FromConvDesc, &Output, Out + OutputLength,
                   &Input, In + InputLength, /*pivotStart=*/NULL,
                   /*pivotSource=*/NULL, /*pivotTarget=*/NULL,
                   /*pivotLimit=*/NULL, /*reset=*/true,
                   /*flush=*/ShouldAutoFlush, &EC);
    if (U_FAILURE(EC)) {
      if (EC == U_BUFFER_OVERFLOW_ERROR &&
          Capacity < std::numeric_limits<size_t>::max())
        HandleOverflow(Capacity, Output, OutputLength, Result);
      else
        // Some other error occured.
        return std::error_code(errno, std::generic_category());
    } else if (U_SUCCESS(EC))
      break;
  } while (U_FAILURE(EC));

  Result.resize(Output - Out);
  return std::error_code();
}

std::error_code CharSetConverterICU::flush() const { return std::error_code(); }

std::error_code
CharSetConverterICU::flush(SmallVectorImpl<char> &Result) const {
  return std::error_code();
}

#elif defined(HAVE_ICONV)
class CharSetConverterIconv : public details::CharSetConverterImplBase {
  iconv_t ConvDesc;

public:
  CharSetConverterIconv(iconv_t ConvDesc) : ConvDesc(ConvDesc) {}

  std::error_code convert(StringRef Source, SmallVectorImpl<char> &Result,
                          bool ShouldAutoFlush) const override;
  std::error_code flush() const override;
  std::error_code flush(SmallVectorImpl<char> &Result) const override;
};

std::error_code CharSetConverterIconv::convert(StringRef Source,
                                               SmallVectorImpl<char> &Result,
                                               bool ShouldAutoFlush) const {
  // Setup the input. Use nullptr to reset iconv state if input length is zero.
  size_t InputLength = Source.size();
  char *Input = InputLength ? const_cast<char *>(Source.data()) : nullptr;
  // Setup the output. We directly write into the SmallVector.
  Result.resize_for_overwrite(Source.size());
  size_t Capacity = Result.capacity();
  char *Output = static_cast<char *>(Result.data());
  size_t OutputLength = Capacity;

  size_t Ret;

  // Handle errors returned from iconv().
  auto HandleError = [&Capacity, &Output, &OutputLength, &Result](size_t Ret) {
    if (Ret == static_cast<size_t>(-1)) {
      // An error occured. Check if we can gracefully handle it.
      if (errno == E2BIG && Capacity < std::numeric_limits<size_t>::max()) {
        HandleOverflow(Capacity, Output, OutputLength, Result);
        return std::error_code();
      } else {
        // Some other error occured.
        return std::error_code(errno, std::generic_category());
      }
    } else {
      // A positive return value indicates that some characters were converted
      // in a nonreversible way, that is, replaced with a SUB symbol. Returning
      // an error in this case makes sure that both conversion routines behave
      // in the same way.
      return std::make_error_code(std::errc::illegal_byte_sequence);
    }
  };

  // Convert the string.
  while ((Ret = iconv(ConvDesc, &Input, &InputLength, &Output, &OutputLength)))
    if (auto EC = HandleError(Ret))
      return EC;
  if (ShouldAutoFlush) {
    while ((Ret = iconv(ConvDesc, nullptr, nullptr, &Output, &OutputLength)))
      if (auto EC = HandleError(Ret))
        return EC;
  }

  // Re-adjust size to actual size.
  Result.resize(Capacity - OutputLength);
  return std::error_code();
}

std::error_code CharSetConverterIconv::flush() const {
  size_t Ret = iconv(ConvDesc, nullptr, nullptr, nullptr, nullptr);
  if (Ret == static_cast<size_t>(-1)) {
    return std::error_code(errno, std::generic_category());
  }
  return std::error_code();
}

std::error_code
CharSetConverterIconv::flush(SmallVectorImpl<char> &Result) const {
  return convert("", Result, /*ShouldAutoFlush=*/true);
}

#endif // HAVE_ICONV
} // namespace

CharSetConverter CharSetConverter::create(text_encoding::id CPFrom,
                                          text_encoding::id CPTo) {

  assert(CPFrom != CPTo && "Text encodings should be distinct");

  ConversionType Conversion;
  if (CPFrom == text_encoding::id::UTF8 && CPTo == text_encoding::id::IBM1047)
    Conversion = UTFToIBM1047;
  else
    Conversion = IBM1047ToUTF;
  std::unique_ptr<details::CharSetConverterImplBase> Converter =
      std::make_unique<CharSetConverterTable>(Conversion);

  return CharSetConverter(std::move(Converter));
}

ErrorOr<CharSetConverter> CharSetConverter::create(StringRef CSFrom,
                                                   StringRef CSTo) {
  std::optional<text_encoding::id> From = getKnownCharSet(CSFrom);
  std::optional<text_encoding::id> To = getKnownCharSet(CSTo);
  if (From && To)
    return create(*From, *To);
#ifdef HAVE_ICU
  UErrorCode EC = U_ZERO_ERROR;
  UConverter *FromConvDesc = ucnv_open(CSFrom.str().c_str(), &EC);
  if (U_FAILURE(EC)) {
    return std::error_code(errno, std::generic_category());
  }
  UConverter *ToConvDesc = ucnv_open(CSTo.str().c_str(), &EC);
  if (U_FAILURE(EC)) {
    return std::error_code(errno, std::generic_category());
  }
  std::unique_ptr<details::CharSetConverterImplBase> Converter =
      std::make_unique<CharSetConverterICU>(FromConvDesc, ToConvDesc);
  return CharSetConverter(std::move(Converter));
#elif defined(HAVE_ICONV)
  iconv_t ConvDesc = iconv_open(CSTo.str().c_str(), CSFrom.str().c_str());
  if (ConvDesc == (iconv_t)-1)
    return std::error_code(errno, std::generic_category());
  std::unique_ptr<details::CharSetConverterImplBase> Converter =
      std::make_unique<CharSetConverterIconv>(ConvDesc);
  return CharSetConverter(std::move(Converter));
#endif
  return std::make_error_code(std::errc::invalid_argument);
}
