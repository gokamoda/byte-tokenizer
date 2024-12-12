from typing import Literal


class ByteTokenizer:
    def __init__(self, method: Literal["utf8", "utf8like"]):
        self.method = method

    def encode(self, text: str) -> list[int]:
        tokenized = []
        for char in text:
            tokenized += self.unicode_to_bytes(ord(char))
        return tokenized

    def decode(self, tokens: list[int]) -> str:
        pass


class UTF8Tokenizer(ByteTokenizer):
    def __init__(self):
        super().__init__("utf8")

    def decode(self, tokens: list[int]) -> str:
        decoded = ""
        i = 0
        while i < len(tokens):
            if tokens[i] < 0b10000000:
                decoded += chr(tokens[i])
                i += 1
            elif tokens[i] < 0b11100000:
                decoded += chr(((tokens[i] & 0b00011111) << 6) + (tokens[i + 1] & 0b00111111))
                i += 2
            elif tokens[i] < 0b11110000:
                decoded += chr(((tokens[i] & 0b00001111) << 12) + ((tokens[i + 1] & 0b00111111) << 6) + (tokens[i + 2] & 0b00111111))
                i += 3
            elif tokens[i] < 0b11111000:
                decoded += chr(
                    ((tokens[i] & 0b00000111) << 18) + ((tokens[i + 1] & 0b00111111) << 12) + ((tokens[i + 2] & 0b00111111) << 6) + (tokens[i + 3] & 0b00111111)
                )
                i += 4
            else:
                raise ValueError("invalid token")
        return decoded

    def unicode_to_bytes(self, codepoint: int) -> list[int]:
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


class UTF8LikeTokenizer(ByteTokenizer):
    def __init__(self):
        super().__init__("utf8like")

    def decode(self, tokens):
        decoded = ""
        i = 0
        while i < len(tokens):
            if tokens[i] < 0b10000000:
                decoded += chr(tokens[i])
                i += 1
            elif tokens[i] < 0b11000000:
                decoded += chr(((tokens[i] & 0b00111111) << 7) + (tokens[i + 1] & 0b01111111))
                i += 2
            elif tokens[i] < 0b11100000:
                decoded += chr(((tokens[i] & 0b00011111) << 13) + ((tokens[i + 1] & 0b00111111) << 7) + (tokens[i + 2] & 0b01111111))
                i += 3
            elif tokens[i] < 0b11110000:
                decoded += chr(
                    ((tokens[i] & 0b00001111) << 18) + ((tokens[i + 1] & 0b00111111) << 13) + ((tokens[i + 2] & 0b00111111) << 7) + (tokens[i + 3] & 0b01111111)
                )
                i += 4
            else:
                raise ValueError("invalid token")
        return decoded

    def unicode_to_bytes(self, codepoint: int) -> list[int]:
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
