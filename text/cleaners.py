""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''


# Regular expression matching whitespace:
import re
from unidecode import unidecode     
#用于将Unicode文本转换为ASCII文本。                   中文文本
#它特别适用于将包含非拉丁字符（如中文、阿拉伯文、俄文等）的文本转换为近似的英文发音。
#这对于文本处理和数据清洗非常有用，尤其是在需要将非英语文本转换为英语发音的场景中。
from phonemizer import phonemize
#Phonemizer 是一个语音合成库，用于将文本转换为音素。    英文文本
#音素是语言中最小的语音单位，phonemize 函数可以处理多种语言的文本，并将其转换为相应的音素序列。
from .numbers import normalize_numbers
_whitespace_re = re.compile(r'\s+')     #\s 是一个特殊的字符类，用于匹配任何空白字符，包括空格、制表符（tab）、换行符等。
                                        #+ 是一个量词，表示匹配前面的字符类一次或多次。

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [       #        
    ('mrs', 'misess'),                                                                  #re.IGNORECASE 是一个编译标志，表示忽略大小写，即在匹配时不区分大写字母和小写字母。
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]

#缩写的拓展，拓展列表定义在_abbreviations中
def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

#text中的numbers.py  这个函数提供了一种有效的方式来将文本中的数字和货币值转换为更自然的语言形式，有助于提高文本的可读性和理解性。
def expand_numbers(text):
    return normalize_numbers(text)

#将输入文本转换为小写。
def lowercase(text):
    return text.lower()

#将文本中的所有连续空白字符（包括空格、制表符、换行符等）替换为单个空格。
def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

#unidecode库，将文本中的非ASCII字符转换为其最接近的ASCII等价物。
def convert_to_ascii(text):
    return unidecode(text)


#下面的cleaners函数是对上面所有函数的单独封装
def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text

def mandarin_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = convert_to_ascii(text)       #把汉字转换为发音最相近的ASCII
    text = lowercase(text)              #转换为小写
    text = collapse_whitespace(text)    #将连续空白字符转换为一个空格
    return text

#vits 的cleaners
def english_cleaners2(text):
  '''Pipeline for English text, including abbreviation expansion. + punctuation + stress'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
  phonemes = collapse_whitespace(phonemes)
  return phonemes