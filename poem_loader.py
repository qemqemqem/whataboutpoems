import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple


class PoemLoader:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.poems: List[Dict[str, Any]] = []
        self.load_poems()

    def load_poems(self) -> None:
        with self.filepath.open('r', encoding='utf-8') as file:
            self.poems = [json.loads(line) for line in file]
            # Remove empty poems
            self.poems = [poem for poem in self.poems if poem['poem'].strip()]

    def get_random_poems(self, count: int = 2) -> List[Dict[str, Any]]:
        return [(poem['author'], poem['title'], poem['poem']) for poem in random.sample(self.poems, count)]

    def get_all_poems(self) -> List[Tuple[str, str, str]]:
        return [(poem['author'], poem['title'], poem['poem']) for poem in self.poems]


if __name__ == "__main__":
    filepath = Path("poems.jsonl")
    loader = PoemLoader(filepath)
    loader.load_poems()

    random_poems = loader.get_random_poems()
    for poem in random_poems:
        print(f"Title: {poem['title']}\nAuthor: {poem['author']}\nPoem: {poem['poem']}\n")
