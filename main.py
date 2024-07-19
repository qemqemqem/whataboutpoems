import os

import requests
from bs4 import BeautifulSoup
import openai
from rich import print
from rich.progress import track

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Download 100 poems
def download_poems():
    print("[bold green]Downloading poems...[/bold green]")
    poems = []
    base_url = "https://www.gutenberg.org/files/"

    for i in range(100):
        url = f"{base_url}{i+1}/{i+1}-0.txt"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            poems.append(text)
            print(f"[green]Downloaded poem {i+1}[/green]")
        else:
            print(f"[red]Failed to download poem {i+1}[/red]")

    return poems

# Step 2: Tokenize poems
def tokenize_poems(poems):
    print("[bold green]Tokenizing poems...[/bold green]")
    # Use OpenAI's tokenizer
    tokenized_poems = [openai.Completion.create(engine="text-davinci-003", prompt=poem, max_tokens=0).choices[0].text.split() for poem in track(poems, description="Tokenizing...")]
    return tokenized_poems

# Step 3: Predict next token and calculate likelihood
def calculate_likelihoods(tokenized_poems):
    print("[bold green]Calculating token likelihoods...[/bold green]")
    likelihoods = []

    for poem in track(tokenized_poems, description="Processing poems..."):
        poem_likelihoods = []
        for i in range(1, len(poem)):
            context = ' '.join(poem[:i])
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=context,
                max_tokens=1,
                logprobs=1
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
    poems = download_poems()

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
