import os
import re
import sys

def process_markdown_file(filepath):
    """
    读取一个 Markdown 文件，处理非代码块、非转义美元区域中数学公式的
    下划线、星号和换行符，并写回文件。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # --- 第 1 步：保护已转义的美元符号 ---
        escaped_dollars = []
        def store_escaped_dollar(match):
            escaped_dollars.append(match.group(0))
            return f"__ESCAPED_DOLLAR_{len(escaped_dollars) - 1}__"
        
        # 匹配 `\$` 或 `\$\$`
        content_no_escapes = re.sub(r'\\(\$\$|\$)', store_escaped_dollar, original_content)

        # --- 第 2 步：保护所有代码块 ---
        code_blocks = []
        def store_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        code_pattern = re.compile(r'```.*?```|`.*?`', re.DOTALL)
        content_no_code = code_pattern.sub(store_code_block, content_no_escapes)

        # --- 第 3 步：在清理后的文本上安全地处理数学公式 ---
        def escape_special_chars_in_math(match):
            """
            转义数学公式内容中的下划线、星号，并处理换行符。
            """
            start_delim, content, end_delim = match.groups()
            
            # 首先，处理下划线：将不是由反斜杠开头的 `_` 替换为 `\_`
            escaped_content = re.sub(r'(?<!\\)_', r'\\_', content)
            
            # 接着，处理星号：将不是由反斜杠开头的 `*` 替换为 `\*`
            escaped_content = re.sub(r'(?<!\\)\*', r'\\*', escaped_content)
            
            # --- 新增：处理 LaTeX 换行符 ---
            # 最后，处理换行符：将 `\\` 替换为 `\\\\`。
            # 这样 Markdown 处理器会将其解析为 `\\`，从而被 KaTeX 正确识别。
            # 使用 re.sub 并用 `r'\\\\'` 匹配精确的两个反斜杠，避免错误替换 `\alpha` 等命令。
            escaped_content = re.sub(r'\\\\', r'\\\\\\\\', escaped_content)
            
            return f"{start_delim}{escaped_content}{end_delim}"

        # 使用非贪婪匹配来查找由 $...$ 或 $$...$$ 包围的数学公式
        math_pattern = re.compile(r'(\$\$|\$)(.*?)(\1)', re.DOTALL)
        processed_content = math_pattern.sub(escape_special_chars_in_math, content_no_code)

        # --- 第 4 步：按相反顺序恢复内容 ---
        # 恢复代码块
        for i, block in reversed(list(enumerate(code_blocks))):
            processed_content = processed_content.replace(f"__CODE_BLOCK_{i}__", block, 1)
        
        # 恢复已转义的美元符号
        for i, dollar in reversed(list(enumerate(escaped_dollars))):
            processed_content = processed_content.replace(f"__ESCAPED_DOLLAR_{i}__", dollar, 1)

        # --- 最后：如果内容有变化，则写回文件 ---
        if processed_content != original_content:
            print(f'  -> Modifying: {filepath}')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(processed_content)

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
    if len(sys.argv) > 1:
        content_dir = sys.argv[1]
    else:
        # 如果没有提供路径，则默认为 'content' 目录
        content_dir = 'content' 
    
    main(content_dir)