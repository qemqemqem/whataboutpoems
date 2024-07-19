import os
from pathlib import Path

import requests
import rich
from bs4 import BeautifulSoup
# import openai
# from openai import OpenAI
import rich
from rich.progress import track
import anthropic
from tqdm import tqdm

from poem_loader import PoemLoader

# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Create a rich console
console = rich.console.Console()

# Step 2: Tokenize poems
def tokenize_poems(poems):
    print("[bold green]Tokenizing poems...[/bold green]")
    # Use OpenAI's tokenizer
    tokenized_poems = []
    for poem in tqdm(poems, desc="Tokenizing..."):
        response = client.completions.create(model="claude-3-5-sonnet-20240620", max_tokens_to_sample=0, prompt=poem)
        # response = client.completions.create(
        #     model="gpt-3.5-turbo-instruct",
        #     prompt=poem,
        #     max_tokens=0
        # )
        tokenized_poems.append(response.choices[0].text.split())
    # tokenized_poems = [client.completions.create(model="gpt-4o-mini", prompt=poem, max_tokens=0).choices[0].text.split() for poem in track(poems, description="Tokenizing...")]
    return tokenized_poems

# Step 3: Predict next token and calculate likelihood
def calculate_likelihoods(tokenized_poems):
    print("[bold green]Calculating token likelihoods...[/bold green]")
    likelihoods = []

    for poem in track(tokenized_poems, description="Processing poems..."):
        poem_likelihoods = []
        for i in range(1, len(poem)):
            context = ' '.join(poem[:i])
            response = client.completions.create(
                model="gpt-4o",
                prompt=context,
                max_tokens=1,
                # logprobs=1
            )
            next_token = poem[i]
            next_token_prob = response['choices'][0]['logprobs']['top_logprobs'][0].get(next_token, float('-inf'))
            poem_likelihoods.append(next_token_prob)
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
    filepath = Path("poems.jsonl")
    loader = PoemLoader(filepath)
    poems_with_attribution = loader.get_all_poems()  # List of (author, title, poem) tuples
    poems = [poem for _, _, poem in poems_with_attribution]

    # Tokenize poems
    tokenized_poems = tokenize_poems(poems)

    # Calculate token likelihoods
    likelihoods = calculate_likelihoods(tokenized_poems)

    # Sort poems based on average correctness
    sorted_poems = sort_poems(poems, likelihoods)

    # Print sorted results
    print("[bold green]Sorted Poems based on Average Token Likelihood[/bold green]")
    for poem, avg_likelihood in sorted_poems:
        print(f"[blue]Poem: {poem[:100]}...[/blue] - [bold green]Average Likelihood: {avg_likelihood:.4f}[/bold green]")

if __name__ == "__main__":
    main()
