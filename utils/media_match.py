import re
import json

def parse_medium_recipe(md_text):
    result = {
        "parameters": {},
        "components": {}
    }

    # 1. 提取 pH 值并计算均值
    ph_match = re.search(r"Final pH:\s*([\d\.\s\-]+)", md_text)
    if ph_match:
        ph_str = ph_match.group(1).strip()
        # 匹配所有数字（整数或浮点数）
        ph_nums = [float(x) for x in re.findall(r"\d+\.?\d*", ph_str)]
        if ph_nums:
            result["parameters"]["ph"] = sum(ph_nums) / len(ph_nums)

    # 2. 按照 "## Component:" 切分
    sections = re.split(r"## Component:\s*", md_text)
    for section in sections[1:]:
        lines = section.strip().split('\n')
        comp_name = lines[0].strip()
        comp_dict = {}

        for line in lines:
            line = line.strip()
            if line.startswith("-") and ":" in line:
                parts = line[1:].split(":", 1)
                reagent = parts[0].strip().lower() # 标准化：转小写
                value_unit = parts[1].strip()

                value_match = re.search(r"^(\d+\.?\d*)", value_unit)
                if value_match:
                    comp_dict[reagent] = float(value_match.group(1))

        if comp_dict:
            result["components"][comp_name] = comp_dict

    return result

def calculate_recipe_loss(gt_dict, pred_dict, missing_penalty=1.0, extra_penalty=0.5):
    """
    计算两个培养基字典之间的 Loss
    :param gt_dict: 真值字典 (Ground Truth)
    :param pred_dict: 预测字典 (Prediction)
    :param missing_penalty: 缺失成分的惩罚权重
    :param extra_penalty: 多出无关成分的惩罚权重
    """
    total_loss = 0.0

    # 1. pH Loss (使用平方误差 MSE)
    ph_gt = gt_dict["parameters"].get("ph")
    ph_pred = pred_dict["parameters"].get("ph")
    if ph_gt is not None and ph_pred is not None:
        total_loss += (ph_gt - ph_pred) ** 2

    # 2. Components Loss
    # 遍历所有组件部分 (如 "Main sol. 119", "Sludge fluid" 等)
    all_sections = set(gt_dict["components"].keys()) | set(pred_dict["components"].keys())

    for section in all_sections:
        gt_comp = gt_dict["components"].get(section, {})
        pred_comp = pred_dict["components"].get(section, {})

        gt_keys = set(gt_comp.keys())
        pred_keys = set(pred_comp.keys())

        # A. 匹配成功的成分：计算数值 Loss (MSE Loss)
        intersect_keys = gt_keys & pred_keys
        for key in intersect_keys:
            v_gt = gt_comp[key]
            v_pred = pred_comp[key]
            # 使用MSE Loss (均方误差)
            mse_loss = (v_gt - v_pred) ** 2
            total_loss += mse_loss

        # B. 缺失的成分 (GT有，Pred没有)
        missing_keys = gt_keys - pred_keys
        total_loss += len(missing_keys) * missing_penalty

        # C. 多出的成分 (Pred有，GT没有)
        extra_keys = pred_keys - gt_keys
        total_loss += len(extra_keys) * extra_penalty

    return total_loss
