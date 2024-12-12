from cpp_module.byte_tokenizer import tokenize, tokenize_utf8_like

if __name__ == "__main__":
    text = "aϴあ"
    tokens = tokenize(text)
    print(tokens)

    tokens = tokenize_utf8_like(text)
    print(tokens)
