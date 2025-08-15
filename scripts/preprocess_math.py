import os
import re
import sys

def escape_underscores_in_math(match):
    """
    一个回调函数，用于 re.sub。
    它接收一个匹配对象，该对象已经通过捕获组分离了定界符和内容。
    """
    # match.group(1) 是开头的定界符 ($$ 或 $)
    # match.group(2) 是公式的实际内容
    # match.group(3) 是结尾的定界符 ($$ 或 $)
    start_delim = match.group(1)
    content = match.group(2)
    end_delim = match.group(3)

    # 核心逻辑：只替换内容中前面不是反斜杠的下划线
    # 这个负向先行断言 `(?<!\\)` 写得非常好，我们继续保留
    escaped_content = re.sub(r'(?<!\\)_', r'\\_', content)

    # 将处理后的内容与原来的定界符重新组合起来
    return f"{start_delim}{escaped_content}{end_delim}"


def process_markdown_file(filepath):
    """
    读取一个 Markdown 文件，处理其中数学公式的下划线，并写回文件。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # 优化后的正则表达式：
        # 使用捕获组来分离定界符和内容。
        # (\$\$|\$) 捕获开始的 $$ 或 $
        # (.*?)    捕获中间的所有内容（非贪婪）
        # (\$\$|\$) 捕获结束的 $$ 或 $
        # 使用 re.DOTALL 使得 `.` 可以匹配换行符，这对于块级公式很重要
        math_pattern = re.compile(r'(\$\$|\$)(.*?)(\1)', re.DOTALL)
        
        # 使用 re.sub 和回调函数进行智能替换
        new_content = math_pattern.sub(escape_underscores_in_math, original_content)

        # 如果内容有变化，则写回文件
        if new_content != original_content:
            print(f'  -> Modifying: {filepath}')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

def main(directory):
    """
    遍历指定目录下的所有 .md 文件并进行处理。
    """
    print(f"Starting math pre-processing in directory: '{directory}'...")
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        sys.exit(1)
        
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                process_markdown_file(os.path.join(root, file))
    
    print("Processing complete.")

if __name__ == '__main__':
    # Zola 的内容通常在 'content' 目录下
    # 您也可以通过命令行参数传入目录，例如：python your_script.py path/to/your/content
    if len(sys.argv) > 1:
        content_dir = sys.argv[1]
    else:
        content_dir = 'content'
    
    main(content_dir)