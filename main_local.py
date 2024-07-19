from pathlib import Path

import requests
from bs4 import BeautifulSoup
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import rich
from rich.progress import track
from rich.markdown import Markdown

from poem_loader import PoemLoader


console = rich.console.Console()

# Step 2: Tokenize poems
def tokenize_poems(poems, tokenizer):
    console.print("[bold green]Tokenizing poems...[/bold green]")
    tokenized_poems = [tokenizer.encode(poem) for poem in track(poems, description="Tokenizing...")]
    return tokenized_poems

# Step 3: Predict next token and calculate likelihood
def calculate_likelihoods(tokenized_poems, model, tokenizer):
    console.print(Markdown("[bold green]Calculating token likelihoods...[/bold green]"))
    model.eval()
    likelihoods = []

    with torch.no_grad():
        for poem in track(tokenized_poems, description="Processing poems..."):
            poem_likelihoods = []
            for i in range(1, len(poem)):
                context = poem[:i]
                inputs = torch.tensor(context).unsqueeze(0)
                outputs = model(inputs)
                next_token_probs = torch.softmax(outputs.logits[0, -1], dim=-1)
                correct_token_prob = next_token_probs[poem[i]].item()
                poem_likelihoods.append(correct_token_prob)
            likelihoods.append(poem_likelihoods)

    return likelihoods

# Step 4: Sort poems based on average correctness
def sort_poems(poems, likelihoods):
    avg_likelihoods = [sum(poem_likelihoods) / len(poem_likelihoods) for poem_likelihoods in likelihoods]
    sorted_poems = sorted(zip(poems, avg_likelihoods), key=lambda x: x[1], reverse=True)
    return sorted_poems

# Main function
def main():
    # Download poems
    console.print(Markdown("# Downloading Poems"))
    filepath = Path("poems.jsonl")
    loader = PoemLoader(filepath)
    poems_with_attribution = loader.get_all_poems()  # List of (author, title, poem) tuples
    poems = [poem for _, _, poem in poems_with_attribution]

    poems = ["Many years later, as he faced the firing squad, Colonel Aureliano Buendía was to remember that distant afternoon when his father took him to discover ice."]

    # Initialize tokenizer and model
    console.print(Markdown("# Initializing Tokenizer and Model"))
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Tokenize poems
    console.print(Markdown("# Tokenizing Poems"))
    tokenized_poems = tokenize_poems(poems, tokenizer)

    # Calculate token likelihoods
    console.print(Markdown("# Downloading Poems"))
    likelihoods = calculate_likelihoods(tokenized_poems, model, tokenizer)
    print(likelihoods)

    # Sort poems based on average correctness
    console.print(Markdown("# Sorting"))
    sorted_poems = sort_poems(poems, likelihoods)

    # Print sorted results
    console.print(Markdown("# Sorted Poems Based on Average Token Likelihood"))
    for poem, avg_likelihood in sorted_poems:
        console.print(f"[blue]Poem: {poem[:100]}...[/blue] - [bold green]Average Likelihood: {avg_likelihood:.4f}[/bold green]")

if __name__ == "__main__":
    main()
