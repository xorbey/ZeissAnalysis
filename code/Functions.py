# This script holds all functions used in the Eminem and Zeiss analysis notebook

import re
from string import punctuation
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')
wordnet_lemmatizer = WordNetLemmatizer()

snowball_stemmer = SnowballStemmer('english')

def to_lower(text:str):
    """
    Casts a string by applying basic string functions
    Args:
        text (str): a string that should be casted to lower case letters

    Returns:
        _type_: the lower case string
    """
    return text.lower()

# Contraction dictionary mapping abbreviated phrases to written out phrases
contractions_dict = {     
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i ain't": "i am not",
"i'd": "i had",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it had",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "iit will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that had",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there had",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they had",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you ain't": "you are not",
"you'd": "you had",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have",
"gonna": "going to",
"wanna": "want to"
}

def expand_contractions(text:str, contractions_dict:dict)->str:
    """
    Replaces abbreviated phrases by proper phrases by mappings in a dictionary
    Args:
        text (str):The text with abbreviated phrases
        contractions_dict (dict): The dictionary holding the mapping between abbreviated and actual phrases

    Returns:
        str: The text with written out phrases 
    """
    for key in contractions_dict.keys():
        if key in text:
            start = text.find(key)
            length = len(key)
            returntext = text[:start] + contractions_dict[key] + text[start+length:]
            return returntext
        else:
            return text


def main_contraction(text:string):
    """ Function to map abbreviated strings to written-out strings by calling the expand_contractions function

    Args:
        text (string): The text with abbreviated phrases

    Returns:
        str: Returns the written out string
    """
    text = expand_contractions(text, contractions_dict)
    return text

# 3: to remove number
def remove_numbers(text:str)->str:
    """Removes numbers from a string

    Args:
        text (str): text with numbers

    Returns:
        str: text without numbers
    """
    output = ''.join(c for c in text if not c.isdigit())
    return output

# 4: remove punctuation
def remove_punct(text:str)->str:
    """Removes punctuation from a string

    Args:
        text (str): text with punctuation

    Returns:
        str: text without punctuation
    """
    return ''.join(c for c in text if c not in punctuation)

# 5: remove whitespace
def to_strip(text:str)->str:
    """Removes larger whitespaces

    Args:
        text (str): text with whitespaces

    Returns:
        str: text with spaces
    """
    return " ".join(text.split())


def remove_stopwords(sentence:str)->str:
    """Removes stopwords present in a a list 

    Args:
        sentence (str): text with stopwords

    Returns:
        str: text without stopwords
    """
    stop_words = stopwords.words('english')
    return ' '.join([w for w in nltk.word_tokenize(sentence) if not w in stop_words])


def lemmatize(text:str)->str:
    """Lemmatize string by using wordnet

    Args:
        text (str): initial text

    Returns:
        str: lemmatized text
    """
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    return " ".join(lemmatized_word)