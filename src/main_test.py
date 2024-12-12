from cpp_module.byte_tokenizer import tokenize, tokenize_utf8_like

from tokenizer import tokenize as py_tokenize
from utils.mylogger import init_logging

LOG_PATH = "test.log"
logger = init_logging(__name__, LOG_PATH, clear=True)


def test_utf8():
    text = "aϴあ𪚲"


    python_result = py_tokenize(text, "utf8")
    cpp_result = tokenize(text)

    logger.info(f"python_result: {python_result}")
    logger.info(f"cpp_result: {cpp_result}")

    assert python_result == cpp_result


def test_utf8like():
    text = "aϴあ\U0010AAAA"

    python_result = py_tokenize(text, "utf8like")
    cpp_result = tokenize_utf8_like(text)

    logger.info(f"python_result: {python_result}")
    logger.info(f"cpp_result: {cpp_result}")

    assert python_result == cpp_result


if __name__ == "__main__":
    test_utf8()
    test_utf8like()
