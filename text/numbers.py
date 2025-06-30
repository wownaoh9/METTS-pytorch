""" from https://github.com/keithito/tacotron """

#inflect: 用于将数字转换为单词。  re: 用于正则表达式匹配和替换
import inflect
import re


_inflect = inflect.engine()                             #这行代码初始化了一个 inflect 引擎，inflect 是一个Python库，用于处理单词的变形，如单数变复数、比较级和最高级等。
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")  #匹配包含千位分隔符的整数。例如，"1,000" 或 "123,456"。它匹配以数字开头，后面跟着一个或多个数字和逗号的组合，最后以数字结尾的序列。
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")    #匹配包含小数点的十进制数。例如，"0.99" 或 "100.5"。它匹配一个或多个数字，后面跟着一个小数点和一个或多个数字。
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")           #匹配英镑货币格式。例如，"£1,000" 或 "£1000"。它匹配一个英镑符号 "£"，后面跟着零个或多个数字和逗号的组合，但至少有一个数字
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")       #匹配美元货币格式。例如，"$1,000.50" 或 "$100"。它匹配一个美元符号 "$"，后面跟着零个或多个数字、小数点和逗号的组合，但至少有一个数字。
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")        #匹配序数词。例如，"1st"、"2nd"、"3rd" 或 "4th"。它匹配一个或多个数字，后面跟着一个序数后缀（"st"、"nd"、"rd" 或 "th"）。
_number_re = re.compile(r"[0-9]+")                      #匹配任何整数。例如，"123" 或 "42"。它匹配一个或多个连续的数字。


# 将逗号替换成空格
def _remove_commas(m):
    return m.group(1).replace(",", "")

#将小数点转换为 "point"
def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")

 #将美元值转换为口语形式（如 "$10.99" 转换为 "ten dollars ninety-nine cents"）。
def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"

#将序数词转换为完整的序数词形式（如 "21" 转换为 "twenty-first"）。
def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


#将数字转换为单词形式（如 "123" 转换为 "one hundred twenty-three"）。
def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(
                num, andword="", zero="oh", group=2
            ).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")

#主函数：这个函数提供了一种有效的方式来将文本中的数字和货币值转换为更自然的语言形式，有助于提高文本的可读性和理解性。
def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text
