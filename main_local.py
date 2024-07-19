from pathlib import Path

import requests
from bs4 import BeautifulSoup
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import rich
from rich.progress import track
from rich.markdown import Markdown
from rich.text import Text

from poem_loader import PoemLoader

console = rich.console.Console()

class Poem:
    def __init__(self, author, title, text):
        self.author = author
        self.title = title
        self.text = text
        self.tokens = None
        self.likelihoods = None

def tokenize(poem, tokenizer):
    poem.tokens = tokenizer.encode(poem.text)

def calculate_likelihoods(poem, model, tokenizer):
    model.eval()
    poem.likelihoods = []
    with torch.no_grad():
        for i in range(1, len(poem.tokens)):
            context = poem.tokens[:i]
            inputs = torch.tensor(context).unsqueeze(0)
            outputs = model(inputs)
            next_token_probs = torch.softmax(outputs.logits[0, -1], dim=-1)
            correct_token_prob = next_token_probs[poem.tokens[i]].item()
            # console.print(f"Computing Likelihood. Context: {context}, Next Token: {tokenizer.decode([poem.tokens[i]])}, Probability: {correct_token_prob}")
            poem.likelihoods.append(correct_token_prob)

def display_colored_tokens(poem, tokenizer):
    token_texts = [tokenizer.decode([token]) for token in poem.tokens]
    for token_text, likelihood in zip(token_texts, poem.likelihoods):
        red = int((1 - likelihood) * 255)
        green = int(likelihood * 255)
        color = f"#{red:02x}{green:02x}00"
        text = Text(token_text, style=f"on {color}")
        console.print(text, end='')
    console.print()  # Newline after printing all tokens

def print_details(poem, tokenizer):
    console.print(Markdown(f"## {poem.title} by {poem.author}"))
    console.print(f"[blue]{poem.text}[/blue]")
    display_colored_tokens(poem, tokenizer)
    if poem.likelihoods:
        avg_likelihood = sum(poem.likelihoods) / len(poem.likelihoods)
        console.print(f"[bold green]Average Likelihood: {avg_likelihood:.4f}[/bold green]")

# Main function
def main():
    # Download poems
    # console.print(Markdown("# Downloading Poems"))
    filepath = Path("poems.jsonl")
    loader = PoemLoader(filepath)
    poems_with_attribution = loader.get_random_poems(3)
    poems = [Poem(author, title, " ".join(poem[:100].split())) for author, title, poem in poems_with_attribution]
    poems.append(Poem("Gabriel Garcia Marquez", "One Hundred Years of Solitude", "Many years later, as he faced the firing squad, Colonel Aureliano Buendía was to remember that distant afternoon when his father took him to discover ice."))
    poems.append(Poem("Shakespeare", "To Be or Not to Be", "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles"))

    # Initialize tokenizer and model
    # console.print(Markdown("# Initializing Tokenizer and Model"))
    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
    model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')

    # Tokenize poems
    # console.print(Markdown("# Tokenizing Poems"))
    for poem in poems:
        tokenize(poem, tokenizer)

    # Calculate token likelihoods
    console.print(Markdown("# Calculating Token Likelihoods"))
    for poem in poems:
        console.print(f"Calculating likelihood of [blue]{poem.title}[/] by [blue]{poem.author}[/]...")
        calculate_likelihoods(poem, model, tokenizer)
        display_colored_tokens(poem, tokenizer)

    # Sort poems based on average correctness
    console.print(Markdown("# Sorting"))
    sorted_poems = sorted(poems, key=lambda p: sum(p.likelihoods) / len(p.likelihoods), reverse=False)

    # Print sorted results
    console.print(Markdown("# Sorted Poems Based on Average Token Likelihood"))
    for poem in sorted_poems:
        print_details(poem, tokenizer)

    console.print(Markdown("# Just the Token Probs, in Order"))
    for poem in sorted_poems:
        display_colored_tokens(poem, tokenizer)

if __name__ == "__main__":
    main()