#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;


std::vector<int> tokenize(const std::string& text, int offset = 0) {
    // UTF-8エンコード済みのバイト列として扱う
    const auto& bytes = reinterpret_cast<const unsigned char*>(text.data());
    size_t length = text.size();

    std::vector<int> result;
    result.reserve(length);

    for (size_t i = 0; i < length; ++i) {
        result.push_back(bytes[i] + offset);
    }
    return result;
}

std::vector<int> tokenize_utf8_like_v1(const std::string& text, int offset = 0) {
    std::vector<int> result;
    const auto& bytes = reinterpret_cast<const unsigned char*>(text.data());

    //// convert multi-byte characters to code points

    // prepare buffer for list of codepoints (max length is text.size())
    std::vector<int> codepoints;
    codepoints.reserve(text.size());

    // initialize codepoint buffer with zeros
    for (size_t i = 0; i < text.size(); ++i) {
        codepoints.push_back(0);
    }

    // iterate over the text
    // i: index of the current byte
    // j: index of the current codepoint
    size_t i = 0;
    size_t num_char = 0;
    for (; i < text.size(); ++num_char) {
        // get the number of bytes in the current character
        unsigned char c = bytes[i];
        int codepoint = 0;
        int n_bytes = 0;
        if (c < 0x80) {
            n_bytes = 1;
            codepoint = c;
        } else if (c < 0xE0) {
            n_bytes = 2;
            codepoint = (c & 0x1F);
        } else if (c < 0xF0) {
            n_bytes = 3;
            codepoint = (c & 0x0F);
        } else if (c < 0xF8) {
            n_bytes = 4;
            codepoint = (c & 0x07);
        } else if (c < 0xFC) {
            n_bytes = 5;
        } else if (c < 0xFE) {
            n_bytes = 6;
        }

        // calculate the codepoint
        for (int k = 1; k < n_bytes; ++k) {
            codepoint = (codepoint << 6) | (bytes[i + k] & 0x3F);
        }
        // store the codepoint
        codepoints[num_char] = codepoint;

        // move to the next character
        i += n_bytes;
    }
    

    // customized byte tokenization

    // iterate over the codepoints
    for (size_t j = 0; j < num_char; ++j) {
        
        // byte head is 0
        if (codepoints[j] < 128) 
        {
            result.push_back(codepoints[j] + offset);
        }

        // byte head is 10 for first 6 bits and 0 for next bit
        else if (codepoints[j]<8192) 
        {
            result.push_back((codepoints[j] >> 7) + 128 + offset);
            result.push_back((codepoints[j] & 127) + offset);
        }

        // byte head is 110 , 10, and 0
        else if(codepoints[j]<262144)
        {
            result.push_back((codepoints[j] >> 13) + 192 + offset);
            result.push_back(((codepoints[j] >> 7) & 0x3F) + 128 + offset);
            result.push_back((codepoints[j] & 0x7F) + offset);
        }

        // byte head is 1110, 110, 10 and 0
        else if(codepoints[j]<4194304)
        {
            result.push_back((codepoints[j] >> 18) + 224 + offset);
            result.push_back(((codepoints[j] >> 13) & 0x1F) + 192 + offset);
            result.push_back(((codepoints[j] >> 7) & 0x3F) + 128 + offset);
            result.push_back((codepoints[j] & 0x7F) + offset);
        }
    }
    return result;
}


std::vector<int> tokenize_utf8_like_v2(const std::string& text, int offset = 0) {
    std::vector<int> result;
    const auto& bytes = reinterpret_cast<const unsigned char*>(text.data());

    //// convert multi-byte characters to code points

    // prepare buffer for list of codepoints (max length is text.size())
    std::vector<int> codepoints;
    codepoints.reserve(text.size());

    // initialize codepoint buffer with zeros
    for (size_t i = 0; i < text.size(); ++i) {
        codepoints.push_back(0);
    }

    // iterate over the text
    // i: index of the current byte
    // j: index of the current codepoint
    size_t i = 0;
    size_t num_char = 0;
    for (; i < text.size(); ++num_char) {
        // get the number of bytes in the current character
        unsigned char c = bytes[i];
        int codepoint = 0;
        int n_bytes = 0;
        if (c < 0x80) {
            n_bytes = 1;
            codepoint = c;
        } else if (c < 0xE0) {
            n_bytes = 2;
            codepoint = (c & 0x1F);
        } else if (c < 0xF0) {
            n_bytes = 3;
            codepoint = (c & 0x0F);
        } else if (c < 0xF8) {
            n_bytes = 4;
            codepoint = (c & 0x07);
        } else if (c < 0xFC) {
            n_bytes = 5;
        } else if (c < 0xFE) {
            n_bytes = 6;
        }

        // calculate the codepoint
        for (int k = 1; k < n_bytes; ++k) {
            codepoint = (codepoint << 6) | (bytes[i + k] & 0x3F);
        }
        // store the codepoint
        codepoints[num_char] = codepoint;

        // move to the next character
        i += n_bytes;
    }
    

    // customized byte tokenization

    // iterate over the codepoints
    for (size_t j = 0; j < num_char; ++j) {

        int b1 = codepoints[j] % 127;
        b1 += 1;
        codepoints[j] /= 127;

        if (codepoints[j] == 0)
        {
            result.push_back(b1);
            continue;   
        }

        int b2 = codepoints[j] % (64 - 8);
        b2 += 128 + 8;
        codepoints[j] /= (64 - 8);

        if (codepoints[j] == 0)
        {
            result.push_back(b2);
            result.push_back(b1);
            continue;
        }

        int b3 = codepoints[j] % (32 - 8);
        b3 += 128 + 64 + 8;
        codepoints[j] /= (32 - 8);

        if (codepoints[j] == 0)
        {
            result.push_back(b3);
            result.push_back(b2);
            result.push_back(b1);
            continue;
        }

        int b4 = codepoints[j] % (32 - 25);
        b4 += 128 + 64 + 32 + 25;
        codepoints[j] /= (32 - 25);

        if (codepoints[j] == 0)
        {
            result.push_back(b4);
            result.push_back(b3);
            result.push_back(b2);
            result.push_back(b1);
            continue;
        }

        throw std::runtime_error("Invalid codepoint");
    }
    return result;
}


PYBIND11_MODULE(byte_tokenizer, m) {
    m.doc() = "Tokenizer module implemented in C++ with pybind11";
    m.def("tokenize", &tokenize, py::arg("text"), py::arg("offset") = 0);
    m.def("tokenize_utf8_like_v1", &tokenize_utf8_like_v1, py::arg("text"), py::arg("offset") = 0);
    m.def("tokenize_utf8_like_v2", &tokenize_utf8_like_v2, py::arg("text"), py::arg("offset") = 0);
}