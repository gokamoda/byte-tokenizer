from cpp_module.byte_tokenizer import tokenize, tokenize_utf8_like_v1, tokenize_utf8_like_v2
from tokenizer import UTF8Tokenizer

if __name__ == "__main__":
    text = "aϴ\nあ"
    tokens = tokenize(text)
    print(tokens)

    tokenizer = UTF8Tokenizer()
    print(tokenizer.decode(tokens))

    tokens = tokenize_utf8_like_v1(text)
    print(tokens)

    tokens = tokenize_utf8_like_v2(text)
    print(tokens)
