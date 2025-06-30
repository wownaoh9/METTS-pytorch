""" from https://github.com/keithito/tacotron """
#返回单词的音素

import re


valid_symbols = [
    "AA",
    "AA0",
    "AA1",
    "AA2",
    "AE",
    "AE0",
    "AE1",
    "AE2",
    "AH",
    "AH0",
    "AH1",
    "AH2",
    "AO",
    "AO0",
    "AO1",
    "AO2",
    "AW",
    "AW0",
    "AW1",
    "AW2",
    "AY",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "EH0",
    "EH1",
    "EH2",
    "ER",
    "ER0",
    "ER1",
    "ER2",
    "EY",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH",
    "IH0",
    "IH1",
    "IH2",
    "IY",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OW0",
    "OW1",
    "OW2",
    "OY",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UH0",
    "UH1",
    "UH2",
    "UW",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]

#集合是一个无序且不重复元素集，可以用于快速成员检查、去除重复项以及集合运算（如并集、交集、差集）
_valid_symbol_set = set(valid_symbols)


class CMUDict:
    """Thin wrapper around CMUDict data. http://www.speech.cs.cmu.edu/cgi-bin/cmudict"""

    def __init__(self, file_or_path, keep_ambiguous=True):          #self 是类的实例，file_or_path 是一个文件路径或者文件对象，keep_ambiguous 是一个布尔值，默认为 True，用于决定是否保留发音不明确的单词
        if isinstance(file_or_path, str):                           #这行代码检查 file_or_path 参数是否是一个字符串                    
            with open(file_or_path, encoding="latin-1") as f:       #拉丁1编码
                entries = _parse_cmudict(f)                         #调用 cmudict.py的_parse_cmudict 函数，解析文件对象，返回一个字典
        else:
            entries = _parse_cmudict(file_or_path)                  #直接将文件对象 file_or_path 传递给 _parse_cmudict 函数
        if not keep_ambiguous:
            entries = {word: pron for word, pron in entries.items() if len(pron) == 1}  #字典推导式：键：word 值：pron for word  逗号后面的式子表示那些只有一个发音的字典键值对将被添加到新字典中
        self._entries = entries                                     #最终得到字典

    def __len__(self):
        return len(self._entries)

    def lookup(self, word):
        """Returns list of ARPAbet pronunciations of the given word."""
        return self._entries.get(word.upper())      #word.upper()：将输入转换成大写字母，因为字典中的键是以大写存储的；get 方法是Python字典的一个方法，它接受一个键（在这个例子中是单词）作为参数，并返回与该键相关联的值。


_alt_re = re.compile(r"\([0-9]+\)")                 #\( 和 \) 分别匹配左圆括号 ( 和右圆括号 )。在正则表达式中，圆括号是特殊字符，用于定义捕获组，所以需要使用反斜杠 \ 进行转义
                                                    #[0-9]：这是一个字符集（character class），用于匹配任何一个介于0到9之间的单个数字。方括号内列出了所有可能匹配的字符
                                                    #+：匹配前面的字符一次或多次。

#返回一个字典，键是单词，值是该单词的音素转录列表。
def _parse_cmudict(file):
    cmudict = {}
    for line in file:
        if len(line) and (line[0] >= "A" and line[0] <= "Z" or line[0] == "'"):     #len(line) 当前行非空
            parts = line.split("  ")                                                #使用两个空格作为分隔符将行分割成单词和它的发音
            word = re.sub(_alt_re, "", parts[0])                                    #re.sub有三个参数，，第一个被第二个替换，第三个是原始字符串，使用正则表达式 _alt_re 替换掉 parts[0]（单词）中的某些字符。
            pronunciation = _get_pronunciation(parts[1])
            if pronunciation:
                if word in cmudict:
                    cmudict[word].append(pronunciation)
                else:
                    cmudict[word] = [pronunciation]
    return cmudict

#返回清理后的音素转录字符串，或者 None（如果包含无效符号）。
def _get_pronunciation(s):
    parts = s.strip().split(" ")                    #s.strip() 去除字符串 s 两端的空白字符（如空格、制表符、换行符等）。    .split(" ") 将清理后的字符串按空格分割成多个部分。这些部分通常表示单词的音素或音标。
    for part in parts:                              
        if part not in _valid_symbol_set:           #检查当前的音素或音标 part 是否在 _valid_symbol_set(有效音素集合) 中
            return None                             #如果发现任何无效符号，则返回 None，表示无法处理该单词
    return " ".join(parts)                          #将清理后的音素转录字符串用空格连接成一个字符串，并返回。




if __name__ == "__main__":
    cmudict = CMUDict

    # 测试 lookup 方法

    path = r"C:\Users\17869\Desktop\111.txt"

    pronunciations = cmudict(path)
    if pronunciations:
        print(f"Pronunciations for '{path}': {pronunciations}")
    else:
        print(f"No pronunciations found for '{path}'.")