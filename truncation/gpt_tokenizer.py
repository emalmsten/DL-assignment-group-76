import tiktoken

class GptTokenizer:
    def __init__(self):
        self.encoding_name = "cl100k_base"

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(self.encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens


