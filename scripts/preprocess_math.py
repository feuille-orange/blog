import os
import re
import sys

def process_markdown_file(filepath):
    """
    读取一个 Markdown 文件，仅处理非代码块、非转义美元区域中数学公式的下划线，并写回文件。
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
        def escape_underscores_in_math(match):
            start_delim, content, end_delim = match.groups()
            escaped_content = re.sub(r'(?<!\\)_', r'\\_', content)
            return f"{start_delim}{escaped_content}{end_delim}"

        math_pattern = re.compile(r'(\$\$|\$)(.*?)(\1)', re.DOTALL)
        processed_content = math_pattern.sub(escape_underscores_in_math, content_no_code)

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
        content_dir = 'content'
    
    main(content_dir)
