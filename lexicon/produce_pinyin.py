# 定义一个空列表来存储所有的字符串
finals = []

# 打开文件，这里假设文件名为"mandarin_pinyin.txt"
with open("mandarin_pinyin.txt", "r") as file:
    # 遍历文件的每一行
    for line in file:
        # 使用split()方法分割每一行的字符串，这里假设字符串之间用空格隔开
        strings = line.strip().split()
        # 将分割后的字符串添加到列表中
        finals.extend(strings)

# 打印结果，或者根据需要进行其他操作
print(finals)

# 将finals列表写入到一个新文件中，每个元素占一行
with open("pinyin.py", "w") as output_file:
    output_file.write("finals = [\n")
    for item in finals:
        output_file.write(f'    "{item}",\n')
    output_file.write("]\n")