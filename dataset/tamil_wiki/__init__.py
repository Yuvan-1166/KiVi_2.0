import os
from bs4 import BeautifulSoup, NavigableString
import nltk
from nltk.tokenize import sent_tokenize
from rich.progress import Progress
from rich.console import Console
from rich.tree import Tree
from rich.live import Live
nltk.download("punkt_tab")
def sentences():
    DIR = os.path.join(os.path.dirname(__file__), "data")
    files = [os.path.join(DIR, f) for f in os.listdir(DIR)]
    with Progress() as progress, Console() as console:
        file_task = progress.add_task("[cyan]Files", total=len(files))
        stack = []
        for f in files:
            if os.path.isfile(f):
                stack.append(os.path.basename(f))
                with open(f, "r", encoding="utf-8") as f:
                    for sentence in process(BeautifulSoup(f.read(), "html.parser"), stack, console):
                        yield sentence
                progress.advance(file_task)
                console.clear()
                console.print(generate_tree(stack))
                stack.pop()


def process(tag, stack: list[str], console):
    stack.append(f"[{tag.name}]: {tag.attrs}")
    # Recursively iterate over child tags
    for child in tag.contents:  # Using tag.contents instead of tag.children
        if isinstance(child, NavigableString):  # Handle NavigableString safely
            if child.strip():  # Ignore empty strings
                for sentence in sent_tokenize(child):
                    sentence = sentence.strip()
                    if sentence:
                        yield sentence
        elif child and child.name:  # Only process tags (not NavigableStrings)
            yield from process(child, stack, console)
    console.clear()
    console.print(generate_tree(stack))
    stack.pop()

def generate_tree(stack: list[str]):
    tree = Tree(stack[0])
    branch = tree
    for s in stack[1:]:
        branch = branch.add(s)
    return tree