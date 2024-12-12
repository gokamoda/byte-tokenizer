from cpp_module.byte_tokenizer import tokenize, tokenize_utf8_like
from tokenizer import UTF8LikeTokenizer, UTF8Tokenizer
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


def test_utf8like():
    text = "aϴあ\U00080000"

    py_tokenizer = UTF8LikeTokenizer()
    python_result = py_tokenizer.encode(text)
    cpp_result = tokenize_utf8_like(text)

    logger.info(f"text: {text}")
    logger.info(f"python_result: {python_result}")
    logger.info(f"cpp_result: {cpp_result}")
    logger.info(py_tokenizer.decode(python_result))

    assert python_result == cpp_result
    assert text == py_tokenizer.decode(python_result)


if __name__ == "__main__":
    test_utf8()
    test_utf8like()
