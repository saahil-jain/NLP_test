import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.
def switch_letters(word):
    percent = 0.10
    if len(word) < 2 or random.random() < 1-percent:
        return word

    positions = random.sample(range(len(word)), 2)

    # Switch the letters at the selected positions
    word_list = list(word)
    word_list[positions[0]], word_list[positions[1]] = word_list[positions[1]], word_list[positions[0]]
    return ''.join(word_list)


def parse_and_switch(sentence):
    words = sentence.split()
    switched_words = [switch_letters(word) for word in words]
    return ' '.join(switched_words)

def generate_typing_errors(sentence):
    percent = 0.01
    sentence = parse_and_switch(sentence)
    alternate_characters = {
        'q': ['w'], 
        'w': ['q', 'e'], 
        'e': ['w', 'r'], 
        'r': ['e', 't'], 
        't': ['r', 'y'], 
        'y': ['t', 'u'], 
        'u': ['y', 'i'], 
        'i': ['u', 'o'], 
        'o': ['i', 'p'], 
        'p': ['o', '['], 
        'a': ['', 's'], 
        's': ['a', 'd'], 
        'd': ['s', 'f'], 
        'f': ['d', 'g'], 
        'g': ['f', 'h'], 
        'h': ['g', 'j'], 
        'j': ['h', 'k'], 
        'k': ['j', 'l'], 
        'l': ['k', ';'], 
        'z': ['x'], 
        'x': ['z', 'c'], 
        'c': ['x', 'v'], 
        'v': ['c', 'b'], 
        'b': ['v', 'n'], 
        'n': ['b', 'm'], 
        'm': ['n', ','], 
        ' ': ['']
    }
    editable_characters = set(alternate_characters.keys())

    transformed_sentence = sentence

    for i, letter in enumerate(sentence):
        if letter in editable_characters and random.random() <= percent:
            replacement_letter = random.choice(alternate_characters[letter])
            transformed_sentence = transformed_sentence[:i] + replacement_letter + transformed_sentence[i + 1:]

    return transformed_sentence

### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

def generate_synonyms(sentence):
    percent = 0.2
    words = sentence.split()
    new_words = [word for word in words]

    max_replacements = 3
    replacements_complete = 0

    for i, word in enumerate(words):

        replacement = word
        if replacements_complete < max_replacements and len(word) > 2:
            if random.random() < percent:
                synonyms = get_synonyms(word)
                if len(synonyms)>0:
                    replacement = random.choice(synonyms)
                    replacements_complete += 1
        new_words[i] = replacement

    return " ".join(new_words)

def switch_US_and_UK_english(sentence):
    percent = 0.2
    if random.random() < 1-percent:
        return sentence
    
    spelling_replacements = {
        'color': 'colour',
        'colour': 'color',
        'favorite': 'favourite',
        'favourite': 'favorite',
        'neighbor': 'neighbour',
        'neighbour': 'neighbor',
        'honor': 'honour',
        'honour': 'honor',
        'humor': 'humour',
        'humour': 'humor',
        'calibre': 'caliber',
        'caliber': 'calibre',
        'centre': 'center',
        'center': 'centre',
        'fibre': 'fiber',
        'fiber': 'fibre',
        'goitre': 'goiter',
        'goiter': 'goitre',
        'litre': 'liter',
        'liter': 'litre',
        'lustre': 'luster',
        'luster': 'lustre',
        'manoeuvre': 'maneuver',
        'maneuver': 'manoeuvre',
        'meagre': 'meager',
        'meager': 'meagre',
        'metre': 'meter',
        'meter': 'metre',
        'mitre': 'miter',
        'miter': 'mitre',
        'nitre': 'niter',
        'niter': 'nitre',
        'ochre': 'ocher',
        'ocher': 'ochre',
        'reconnoitre': 'reconnoiter',
        'reconnoiter': 'reconnoitre',
        'sabre': 'saber',
        'saber': 'sabre',
        'saltpetre': 'saltpeter',
        'saltpeter': 'saltpetre',
        'sepulchre': 'sepulcher',
        'sepulcher': 'sepulchre',
        'sombre': 'somber',
        'somber': 'sombre',
        'spectre': 'specter',
        'specter': 'spectre',
        'theatre': 'theater',
        'theater': 'theatre',
        'titre': 'titer',
        'titer': 'titre'
    }
    word_ending_replacement = {
        'yze': 'yse',
        'yse': 'yze',
        'yzed': 'ysed',
        'ysed': 'yzed',
        'ize': 'ise',
        'ise': 'ize',
        'izer': 'iser',
        'iser': 'izer',
        'ization': 'isation',
        'isation': 'ization',
        'logue': 'log',
        'log': 'logue',
    }
    spelling_replacement_keys = set(spelling_replacements.keys())
    word_ending_keys = set(word_ending_replacement.keys())
    
    words = sentence.split()
    new_words = [word for word in words]
    for i, word in enumerate(words):
        if word in spelling_replacement_keys:
            new_words[i] = spelling_replacements[word]
        else:
            for ending in word_ending_keys:
                if word.endswith(ending):
                    new_words[i] = word[:-len(ending)] + word_ending_replacement[ending]

    return " ".join(new_words)


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    sentences = example["text"].split(". ")
    new_sentences = [sentence for sentence in sentences]

    for i, sentence in enumerate(sentences):
        new_sentence = generate_synonyms(sentence)
        new_sentence = generate_typing_errors(new_sentence)
        new_sentence = switch_US_and_UK_english(new_sentence)
        new_sentences[i] = new_sentence

    example["text"] = ". ".join(new_sentences)
    ##### YOUR CODE ENDS HERE ######

    return example

    ##### YOUR CODE ENDS HERE ######

    return example

# Example usage
# american_sentence = {}
# american_sentence['text'] = "I love the color of your car, it's my favorite. Let's organize a theater event."
# british_sentence = custom_transform(american_sentence)['text']
# print(british_sentence)