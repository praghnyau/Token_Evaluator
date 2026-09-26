import tiktoken

def count_tokens(text: str, model: str = "gemini-1.5-flash") -> int:
    """
    Counts the number of tokens in a string using tiktoken.
    Falls back to a word/character ratio estimation if tiktoken fails.
    """
    if not text:
        return 0
        
    try:
        # standard cl100k_base encoding is close enough for Gemini models
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to general approximation: ~4 characters per token
        # or ~0.75 words per token.
        char_count = len(text)
        word_count = len(text.split())
        approx_tokens = max(int(char_count / 4.0), int(word_count / 0.75), 1)
        return approx_tokens
