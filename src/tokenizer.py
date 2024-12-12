from typing import Literal


def tokenize(text: str, method: Literal["utf8", "utf8like"]) -> list[int]:
    tokenized = []
    for char in text:
        if method == "utf8":
            tokenized += unicode_to_utf8(ord(char))
        elif method == "utf8like":
            tokenized += unicode_to_utf8like(ord(char))
    return tokenized


def unicode_to_utf8(
    codepoint,
) -> list[int]:
    codepoint_bin = f"{codepoint:b}"

    if len(codepoint_bin) <= 7:  # 1byte char
        codepoint_bin = f"{codepoint:07b}"
        bytes_bin = [
            "0" + codepoint_bin,
        ]
    elif len(codepoint_bin) <= 11:  # 2byte char
        codepoint_bin = f"{codepoint:011b}"
        bytes_bin = [
            "110" + codepoint_bin[:5],
            "10" + codepoint_bin[5:],
        ]
    elif len(codepoint_bin) <= 16:  # 3byte char
        codepoint_bin = f"{codepoint:016b}"
        bytes_bin = [
            "1110" + codepoint_bin[:4],
            "10" + codepoint_bin[4:10],
            "10" + codepoint_bin[10:],
        ]
    elif len(codepoint_bin) <= 21:  # 4byte char
        codepoint_bin = f"{codepoint:021b}"
        bytes_bin = [
            "11110" + codepoint_bin[:3],
            "10" + codepoint_bin[3:9],
            "10" + codepoint_bin[9:15],
            "10" + codepoint_bin[15:],
        ]
    else:
        raise ValueError("codepoint is too large")

    return [int(byte, 2) for byte in bytes_bin]


def unicode_to_utf8like(
    codepoint,
) -> list[int]:
    codepoint_bin = f"{codepoint:b}"

    if len(codepoint_bin) <= 7:  # 1byte char
        codepoint_bin = f"{codepoint:07b}"
        bytes_bin = [
            "0" + codepoint_bin,
        ]
    elif len(codepoint_bin) <= 13:  # 2byte char
        codepoint_bin = f"{codepoint:013b}"
        bytes_bin = [
            "10" + codepoint_bin[:6],
            "0" + codepoint_bin[6:],
        ]
    elif len(codepoint_bin) <= 18:  # 3byte char
        codepoint_bin = f"{codepoint:018b}"
        bytes_bin = [
            "110" + codepoint_bin[:5],
            "10" + codepoint_bin[5:11],
            "0" + codepoint_bin[11:],
        ]
    elif len(codepoint_bin) <= 22:  # 4byte char
        codepoint_bin = f"{codepoint:022b}"
        bytes_bin = [
            "1110" + codepoint_bin[:4],
            "110" + codepoint_bin[4:9],
            "10" + codepoint_bin[9:15],
            "0" + codepoint_bin[15:],
        ]
    else:
        raise ValueError("codepoint is too large")

    return [int(byte, 2) for byte in bytes_bin]