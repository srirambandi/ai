"""
Byte Pair Encoding tokenizer based on Andrej Karpathy's Tokenizer video.
https://www.youtube.com/watch?v=zduSFxRajkE
"""


from typing import List, Tuple
from ai.utils import safe_ensure_future, safe_gather


def get_pairs(tokens):
    count = {}
    for pair in zip(tokens, tokens[1:]):
        count[pair] = count.get(pair, 0) + 1

    return count


def merge_pair(tokens: List[int], pair: Tuple[int, int], new_token_id: int):
    new_tokens = []
    i = 0
    token_1, token_2 = pair
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == token_1 and tokens[i + 1] == token_2:
            new_tokens.append(new_token_id)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    
    return new_tokens


class BPETokenizer:
    def __init__(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}

    def train(self, corpus: str, num_of_merges: int = 0, SPECIAL_TOKENS: List[str] = []):
        byte_stream = corpus.encode('utf-8')
        tokens = list(byte_stream)
        for i in range(num_of_merges):
            pairs = get_pairs(tokens)
            top_pair = max(pairs, key=pairs.get)
            new_token_id = 256 + i
            tokens = merge_pair(tokens, top_pair, new_token_id)
            self.merges[top_pair] = new_token_id

        for (token_1, token_2), token_id in self.merges.items():
            self.vocab[token_id] = self.vocab[token_1] + self.vocab[token_2]

        for i in range(len(SPECIAL_TOKENS)):
            it = 256 + num_of_merges + i
            self.vocab[it] = SPECIAL_TOKENS[i]

    def encode(self, texts: List[str]) -> List[List[int]]:
        tasks = []
        for text in texts:
            tasks.append(self._encode_single_sequence(text))

        return safe_ensure_future(safe_gather(*tasks))

    async def _encode_single_sequence(self, text: str) -> List[int]:
        byte_stream = text.encode('utf-8')
        tokens = list(byte_stream)
        # TODO: regex split the tokens for efficient tokenization

        while len(tokens) >= 2:
            pairs = get_pairs(tokens)
            mergeable_pair = min(pairs, key=lambda pair: self.merges.get(pair, float('inf')))
            if mergeable_pair not in self.merges:
                break
            token_id = self.merges[mergeable_pair]
            tokens = merge_pair(tokens, mergeable_pair, token_id)
        
        return tokens

    def decode(self, tokens_lists: List[List[int]]) -> List[str]:
        tasks = []
        for token_seq in tokens_lists:
            tasks.append(self._decode_single_sequence(token_seq))

        return safe_ensure_future(safe_gather(*tasks))

    async def _decode_single_sequence(self, tokens: List[int]) -> str:
        tokens = b"".join(self.vocab[token] for token in tokens)
        text = tokens.decode("utf-8", errors="replace")

        return text
