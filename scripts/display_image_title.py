import os
import re
import sys
import html

def process_markdown_file(filepath):
    """
    读取一个 Markdown 文件，将其中带 title 的图片语法转换为 HTML <figure> 块。
    这个过程会保护代码块中的内容不被修改。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # --- 第 1 步：保护所有代码块 ---
        code_blocks = []
        def store_code_block(match):
            """回调函数：存储代码块并返回一个占位符"""
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        # 匹配多行代码块 (```...```) 和行内代码 (`)
        code_pattern = re.compile(r'```.*?```|`.*?`', re.DOTALL)
        content_no_code = code_pattern.sub(store_code_block, original_content)

        # --- 第 2 步：在清理后的文本上安全地转换图片 ---
        def convert_image_to_figure(match):
            """回调函数：将匹配的图片语法转换为 <figure> HTML"""
            # 从匹配项中提取 alt, src, 和 title
            alt_text = html.escape(match.group(1), quote=True)
            src_path = html.escape(match.group(2), quote=True)
            # group(3) 捕获的是 title 的内容，不包含引号
            caption_text = html.escape(match.group(3), quote=True)
            
            # 构建 HTML 块，保持良好缩进
            return (
                f'<figure>\n'
                f'  <img src="{src_path}" alt="{alt_text}">\n'
                f'  <figcaption>{caption_text}</figcaption>\n'
                f'</figure>'
            )

        # 优化后的正则表达式：
        # 匹配带 title 的图片: ![alt](src "title") 或 ![alt](src 'title')
        # ["'] 匹配双引号或单引号
        # (.*?) 捕获引号内的内容
        image_pattern = re.compile(r'!\[(.*?)\]\((\S+)\s+["\'](.*?)["\']\)')
        processed_content = image_pattern.sub(convert_image_to_figure, content_no_code)

        # --- 第 3 步：按相反顺序恢复代码块 ---
        for i, block in reversed(list(enumerate(code_blocks))):
            processed_content = processed_content.replace(f"__CODE_BLOCK_{i}__", block, 1)

        # --- 最后：如果内容有变化，则写回文件 ---
        if processed_content != original_content:
            print(f'  -> Modifying: {filepath}')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(processed_content)

    except Exception as e:
        print(f"Error processing file {filepath}: {e}", file=sys.stderr)

def main(directory):
    """
    遍历指定目录下的所有 .md 文件并进行处理。
    """
    print(f"Starting image conversion in directory: '{directory}'...")
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.", file=sys.stderr)
        sys.exit(1)
        
    for root, _, files in os.walk(directory):
        for file in files:
            # 支持 .md 和 .markdown 后缀
            if file.endswith(('.md', '.markdown')):
                process_markdown_file(os.path.join(root, file))
    
    print("Image conversion complete.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        content_dir = sys.argv[1]
    else:
        content_dir = 'content'
    
    main(content_dir)