import re
import argparse
from string import punctuation

import yaml
import numpy as np
from g2p_en import G2p
from pypinyin import pinyin, Style

from text import text_to_sequence

def preprocess_mandarin_infer(text):
    lexicon = read_lexicon("lexicon/mandarin_pinyin.txt")

    phones = []
    #用pypinyin得到的拼音
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")
    phones = "{" + " ".join(phones) + "}"
    return phones

def preprocess_english_infer(text):
    text = text.rstrip(punctuation)           

    lexicon = read_lexicon("lexicon/librispeech-lexicon.txt")
    g2p = G2p()

    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)

    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    # phones = phones.replace("}{", " ")
    return phones
def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["zh_lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    return phones

def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)           

    lexicon = read_lexicon(preprocess_config["path"]["en_lexicon_path"])
    g2p = G2p()

    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)

    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    # phones = phones.replace("}{", " ")

    return phones

#pho2seq
def pho2seq(phones):
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return np.array(sequence)



def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default="today is Monday 12 o'clock,when to go to lunch?",
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=False, default="../config/LJSpeech/preprocess.yaml",
        help="path to preprocess.yaml",
    )
    args = parser.parse_args()


    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )

    phones=preprocess_english(args.text,preprocess_config)