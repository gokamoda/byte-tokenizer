from cpp_module.byte_tokenizer import tokenize, tokenize_utf8_like_v1, tokenize_utf8_like_v2
from tokenizer import ByteLMTokenizerV1, ByteLMTokenizerV2, UTF8Tokenizer
from utils.mylogger import init_logging

LOG_PATH = "test.log"
logger = init_logging(__name__, LOG_PATH, clear=True)


def test_utf8():
    text = "aϴあ\U00080000"
    py_tokenizer = UTF8Tokenizer()
    python_result = py_tokenizer.encode(text)
    cpp_result = tokenize(text)

    logger.info(f"text: {text}")
    logger.info(f"python_result: {python_result}")
    logger.info(f"cpp_result: {cpp_result}")
    logger.info(py_tokenizer.decode(python_result))

    assert python_result == cpp_result
    assert text == py_tokenizer.decode(python_result)


def test_utf8like_v1():
    text = "aϴあ\U00080000"

    py_tokenizer = ByteLMTokenizerV1()
    python_result = py_tokenizer.encode(text)
    cpp_result = tokenize_utf8_like_v2(text)

    logger.info(f"text: {text}")
    logger.info(f"python_result: {python_result}")
    logger.info(f"cpp_result: {cpp_result}")
    logger.info(py_tokenizer.decode(python_result))

    assert python_result == cpp_result
    assert text == py_tokenizer.decode(python_result)


def test_utf8like_v2():
    text = "aϴあ\U00080000"



    py_tokenizer = ByteLMTokenizerV2()

    python_result = py_tokenizer.encode(text)
    cpp_result = tokenize_utf8_like_v2(text)


    logger.info(f"text: {text}")
    logger.info(f"python_result: {python_result}")
    logger.info(f"cpp_result: {cpp_result}")
    logger.info(py_tokenizer.decode(python_result))

    assert python_result == cpp_result
    assert text == py_tokenizer.decode(python_result)


if __name__ == "__main__":
    # test_utf8()
    # test_utf8like_v1()
    test_utf8like_v2()

    tokenizer = ByteLMTokenizerV2()
    byte_sets = set(list(range(0, 256)))
    with open(f"encodings_{tokenizer.method}.txt", "w") as f:
        for i in range(int('0x110000', 16)):
            byte_ints = tokenizer.unicode_to_bytes(i)
            byte_bins = [f"{byte_int:08b}" for byte_int in byte_ints]
            f.write(f"{i} {' '.join(byte_bins)}\n")
            byte_sets -= set(byte_ints)

    print(byte_sets)
