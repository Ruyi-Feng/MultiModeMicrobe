import json
import os
import re
import glob
from pathlib import Path
from json_loader import load_json, save_json

def clean_text(text):
    """
    清洗字符串：
    1. 处理 None 值
    2. 将连续的空白字符（换行、制表符、多空格）替换为单个空格
    3. 去除首尾空格
    """
    if text is None:
        return ""
    text = str(text)
    # 替换所有类型的空白符（\n, \t, \r, \f, \v）为一个空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def convert_json_to_markdown(data):
    """
    将单个培养基 JSON 对象转换为 Markdown 格式字符串
    """
    lines = []

    # --- 1. Header Info ---
    # 提取 ID 和 名称
    medium_id = clean_text(data.get("id", "Unknown ID"))
    name = clean_text(data.get("name", "Unknown Medium"))
    lines.append(f"# Medium: {name} (ID: {medium_id})")

    # --- 2. Parameters (关键元数据) ---
    metadata = data.get("metadata", {})
    ph = clean_text(metadata.get("Final pH:", "Not specified"))

    lines.append("\n## Parameters")
    lines.append(f"- Final pH: {ph}")

    # 提取气体环境
    gas_info = data.get("gas_composition", {})
    gas_list = gas_info.get("gas_list", [])
    if gas_list:
        # 清洗气体列表中的每一项
        clean_gases = [clean_text(g) for g in gas_list]
        lines.append(f"- Gas: {', '.join(clean_gases)}")

    # --- 3. Recipe Components (核心配方) ---
    recipes = data.get("recipe", [])

    for sub_recipe in recipes:
        # 子配方标题 (e.g., Main sol., Trace element solution)
        title = clean_text(sub_recipe.get("title", "Component"))
        lines.append(f"\n## Component: {title}")

        # 成分列表
        ingredients = sub_recipe.get("recipe_list", [])
        if ingredients:
            for item in ingredients:
                compound = clean_text(item.get("Compound"))
                amount = clean_text(item.get("Amount"))
                unit = clean_text(item.get("Unit"))

                # 构建行: "- Compound: Amount Unit"
                # 如果没有单位，就不显示单位
                if unit and unit != "-":
                    line = f"- {compound}: {amount} {unit}"
                else:
                    line = f"- {compound}: {amount}"
                lines.append(line)

        # 操作步骤
        steps = sub_recipe.get("step_list", [])
        if steps:
            for step in steps:
                cleaned_step = clean_text(step)
                if cleaned_step:
                    # 使用引用块符号 '>' 标记步骤，便于模型区分成分和动作
                    lines.append(f"> Instructions: {cleaned_step}")

    # 将列表组合成最终的字符串，用换行符连接
    return "\n".join(lines)

def process_folder(input_folder, output_folder):
    """
    批量处理文件夹
    """
    # 确保输出目录存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # 获取所有 json 文件
    json_files = glob.glob(os.path.join(input_folder, "*.json"))

    print(f"Found {len(json_files)} JSON files in '{input_folder}'. Processing...")

    success_count = 0

    for json_file in json_files:
        try:
            # 1. 读取 JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 2. 转换格式
            markdown_content = convert_json_to_markdown(data)

            # 3. 确定输出路径 (更改后缀为 .md)
            base_name = os.path.basename(json_file)
            file_name_without_ext = os.path.splitext(base_name)[0]
            output_path = os.path.join(output_folder, file_name_without_ext + ".md")

            # 4. 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            success_count += 1

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    print(f"Processing complete. {success_count}/{len(json_files)} files converted successfully.")
    print(f"Saved to: {output_folder}")


def generate_media_dict(overview_path, media_md_files_path, save_path):
    media_dict = {}
    overview = load_json(overview_path)
    for item in overview:
        bacdive_id = item["BacDive ID"]
        if "MeidaDive_Media_Paths" in item:
            media_file_paths = item["MeidaDive_Media_Paths"]
            media_md_paths = []
            for media_file_path in media_file_paths:
                _, file_name = os.path.split(media_file_path)
                new_file_path = os.path.join(media_md_files_path, file_name.replace(".json", ".md"))
                media_md_paths.append(new_file_path)
            media_dict[bacdive_id] = media_md_paths
        else:
            media_dict[bacdive_id] = []
            print(f"No MeidaDive_Media_Paths for {bacdive_id}")
    save_json(media_dict, save_path)
    print(f"Generated media dict and saved to {save_path}")
    return media_dict


if __name__ == "__main__":
    input_dir = "C:\\Users\\User\\WorkSpace\\data\\Zhiling\\aaseq_microbe\\examples_methano\\data_files\\media_json_files"   # 存放原始 JSON 的文件夹
    save_path = "C:\\Users\\User\\WorkSpace\\data\\Zhiling\\aaseq_microbe\\examples_methano\\data_files\\media_md_files" # 存放处理后 Markdown 的文件夹

    media_md_files_path = "media_md_files\\"
    overview_path = "C:\\Users\\User\\WorkSpace\\data\\Zhiling\\aaseq_microbe\\examples_methano\\microbex_data_with_eggnog.json"
    save_path = "C:\\Users\\User\\WorkSpace\\data\\Zhiling\\aaseq_microbe\\examples_methano\\data_files\\media_dict.json"
    # process_folder(input_dir, save_path)
    generate_media_dict(overview_path, media_md_files_path, save_path)
