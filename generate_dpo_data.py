#!/usr/bin/env python3
"""
生成 BanFakeNews 数据集的 DPO 训练数据

DPO (Direct Preference Optimization) 需要三元组数据：
- prompt: 输入问题
- chosen: 正确的回答（高质量）
- rejected: 错误的回答（低质量）

这个脚本将生成两种类型的 DPO 数据对：
1. 正确标签 vs 错误标签
2. 详细推理 vs 简短推理
"""

import os
import json
import argparse
import random
from tqdm import tqdm


def generate_dpo_prompt(text, budget=128):
    """
    生成 DPO 训练的 prompt

    Args:
        text: 新闻文本
        budget: token 预算限制

    Returns:
        格式化的 prompt 字符串
    """
    prompt = f"""Q: {text}

Please determine if the above news article is Real or Fake. Analyze the content carefully and provide your reasoning.

Please give the response by strictly following this format: [[label]], for example: Label: [[Real]] or Label: [[Fake]].

Let's think step by step and use less than {budget} tokens:
"""
    return prompt


def generate_chosen_response(label, quality="high"):
    """
    生成高质量的 chosen 回答

    Args:
        label: 真实标签 ('Real' 或 'Fake')
        quality: 回答质量 ('high' 表示详细，'medium' 表示中等)

    Returns:
        chosen 回答字符串
    """
    if label == "Real":
        if quality == "high":
            responses = [
                "After careful analysis, this article demonstrates several characteristics of authentic journalism:\n\n1. **Credible sourcing**: The article cites verifiable sources including official statements and expert opinions.\n2. **Factual consistency**: The claims align with established facts and can be cross-referenced.\n3. **Professional writing**: The language is neutral, objective, and follows journalistic standards.\n4. **Verifiable details**: Specific names, dates, and locations can be independently confirmed.\n5. **Balanced perspective**: Multiple viewpoints are presented without sensationalism.\n\nBased on these indicators, this is authentic reporting.\n\nLabel: [[Real]]",

                "Detailed evaluation reveals this is genuine news:\n\n**Evidence of authenticity:**\n- Multiple credible sources cited (government agencies, established media)\n- Facts can be verified through independent databases\n- Writing style matches professional journalism standards\n- No emotional manipulation or sensational language\n- Timeline and details are internally consistent\n- Author credentials are verifiable\n\nThe article adheres to journalistic integrity and factual reporting.\n\nLabel: [[Real]]"
            ]
        else:  # medium
            responses = [
                "This article shows signs of authentic reporting: credible sources are cited, the facts are verifiable, and the writing maintains objectivity. The information aligns with known events and can be cross-referenced. No red flags for misinformation detected.\n\nLabel: [[Real]]",

                "Analysis indicates genuine journalism: The content includes verifiable claims, proper sourcing, and professional presentation. The narrative is factually consistent and lacks manipulative elements typical of fake news.\n\nLabel: [[Real]]"
            ]
    else:  # Fake
        if quality == "high":
            responses = [
                "This article exhibits multiple red flags indicating misinformation:\n\n1. **Lack of credible sources**: No verifiable sources or citations provided.\n2. **Factual inconsistencies**: Claims contradict established facts and timelines.\n3. **Emotional manipulation**: Highly sensational language designed to provoke outrage.\n4. **Unverifiable claims**: Extraordinary assertions without supporting evidence.\n5. **Logical fallacies**: The narrative contains internal contradictions.\n6. **Suspicious origin**: The source lacks credibility or transparency.\n\nThese characteristics are hallmarks of fabricated content designed to mislead.\n\nLabel: [[Fake]]",

                "Critical analysis reveals this is misinformation:\n\n**Red flags identified:**\n- No credible attribution or sources\n- Factual claims contradict verified information\n- Highly emotional and manipulative language\n- Designed to provoke rather than inform\n- Contains logical inconsistencies\n- Lacks journalist byline or editorial oversight\n\nThis content appears intentionally fabricated to spread false information.\n\nLabel: [[Fake]]"
            ]
        else:  # medium
            responses = [
                "This article shows clear signs of fake news: no credible sources, sensational language, unverifiable claims, and factual inconsistencies. These red flags indicate the content is likely fabricated or heavily manipulated.\n\nLabel: [[Fake]]",

                "Multiple indicators suggest this is misinformation: lack of proper sourcing, emotional manipulation tactics, contradiction of established facts, and extraordinary claims without evidence. This is not credible journalism.\n\nLabel: [[Fake]]"
            ]

    return random.choice(responses)


def generate_rejected_response(label, error_type="wrong_label"):
    """
    生成低质量的 rejected 回答

    Args:
        label: 真实标签 ('Real' 或 'Fake')
        error_type: 错误类型
            - 'wrong_label': 标签完全错误
            - 'weak_reasoning': 推理薄弱
            - 'no_explanation': 缺乏解释

    Returns:
        rejected 回答字符串
    """
    wrong_label = "Fake" if label == "Real" else "Real"

    if error_type == "wrong_label":
        # 完全错误的标签
        if wrong_label == "Real":
            responses = [
                "The article appears credible with proper sources. Label: [[Real]]",
                "This seems to be authentic reporting. Label: [[Real]]",
                "No obvious signs of misinformation detected. Label: [[Real]]"
            ]
        else:
            responses = [
                "This looks like fake news to me. Label: [[Fake]]",
                "The content seems suspicious and unreliable. Label: [[Fake]]",
                "Multiple red flags suggest this is fabricated. Label: [[Fake]]"
            ]

    elif error_type == "weak_reasoning":
        # 推理薄弱但标签正确
        responses = [
            f"I think this is probably {label.lower()}. Label: [[{label}]]",
            f"Based on my feeling, this appears to be {label.lower()} news. Label: [[{label}]]",
            f"It seems like {label.lower()} news to me. Label: [[{label}]]"
        ]

    else:  # no_explanation
        # 缺乏解释
        responses = [
            f"Label: [[{label}]]",
            f"Answer: [[{label}]]",
            f"Classification: [[{label}]]"
        ]

    return random.choice(responses)


def load_csv_data(csv_path, label):
    """
    从 CSV 文件加载数据

    Args:
        csv_path: CSV 文件路径
        label: 该文件的标签 ('Real' 或 'Fake')

    Returns:
        数据列表
    """
    import pandas as pd

    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except:
        df = pd.read_csv(csv_path, encoding='latin-1')

    # 尝试找到文本列
    possible_text_columns = ['text', 'content', 'article', 'news', 'headline',
                             'title', 'body', 'full_text', 'Text', 'Content']

    text_column = None
    for col in possible_text_columns:
        if col in df.columns:
            text_column = col
            break

    if text_column is None:
        text_column = df.columns[0]

    data = []
    for _, row in df.iterrows():
        text = str(row[text_column]).strip()
        if len(text) > 50:  # 过滤太短的文本
            data.append({
                'text': text,
                'label': label
            })

    return data


def generate_dpo_data(args):
    """
    生成 DPO 训练数据
    """
    print("=" * 60)
    print("生成 BanFakeNews DPO 训练数据")
    print("=" * 60)

    # 加载训练集数据
    print("\n加载训练集数据...")
    train_data = []

    # 真实新闻
    authentic_path = os.path.join(args.data_dir, 'Authentic-48K.csv')
    if os.path.exists(authentic_path):
        authentic_data = load_csv_data(authentic_path, 'Real')
        print(f"✓ 加载真实新闻: {len(authentic_data)} 条")
        train_data.extend(authentic_data)

    # 假新闻
    fake_path = os.path.join(args.data_dir, 'Fake-1K.csv')
    if os.path.exists(fake_path):
        fake_data = load_csv_data(fake_path, 'Fake')
        print(f"✓ 加载假新闻: {len(fake_data)} 条")
        train_data.extend(fake_data)

    if len(train_data) == 0:
        print("❌ 错误: 没有找到训练数据！")
        return

    # 打乱数据
    random.shuffle(train_data)

    # 限制数据量
    if args.max_samples and args.max_samples < len(train_data):
        train_data = train_data[:args.max_samples]
        print(f"✓ 限制为前 {args.max_samples} 条样本")

    print(f"\n总训练样本: {len(train_data)} 条")

    # 生成 DPO 数据
    print("\n生成 DPO 训练数据...")
    dpo_data = []

    for item in tqdm(train_data, desc="生成 DPO 数据"):
        # 随机选择预算
        budget = random.choice(args.budgets)

        # 生成 prompt
        prompt = generate_dpo_prompt(item['text'], budget)

        # 生成 chosen（高质量回答）
        quality = random.choice(['high', 'medium'])
        chosen = generate_chosen_response(item['label'], quality)

        # 生成 rejected（低质量回答）
        # 70% 使用错误标签，30% 使用薄弱推理
        if random.random() < 0.7:
            error_type = 'wrong_label'
        else:
            error_type = random.choice(['weak_reasoning', 'no_explanation'])

        rejected = generate_rejected_response(item['label'], error_type)

        dpo_data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'label': item['label'],
            'budget': budget,
            'error_type': error_type
        })

    # 保存训练数据
    output_path = args.output_path
    print(f"\n保存 DPO 训练数据到: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dpo_data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"✓ 已保存 {len(dpo_data)} 条 DPO 训练数据")

    # 保存统计信息
    stats = {
        'total_samples': len(dpo_data),
        'real_samples': sum(1 for x in dpo_data if x['label'] == 'Real'),
        'fake_samples': sum(1 for x in dpo_data if x['label'] == 'Fake'),
        'error_types': {
            'wrong_label': sum(1 for x in dpo_data if x.get('error_type') == 'wrong_label'),
            'weak_reasoning': sum(1 for x in dpo_data if x.get('error_type') == 'weak_reasoning'),
            'no_explanation': sum(1 for x in dpo_data if x.get('error_type') == 'no_explanation')
        },
        'budgets': args.budgets
    }

    print("\n" + "=" * 60)
    print("统计信息:")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="生成 BanFakeNews DPO 训练数据")

    # 数据路径
    parser.add_argument("--data_dir", default="./data/BanFakeNews",
                        help="BanFakeNews CSV 文件目录")
    parser.add_argument("--output_path", default="./data/banfakenews_train_dpo.jsonl",
                        help="DPO 训练数据输出路径")

    # 数据生成选项
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大训练样本数（None 表示使用全部）")
    parser.add_argument("--budgets", type=int, nargs='+', default=[64, 128, 256],
                        help="token 预算选项列表")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子
    random.seed(args.seed)

    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"❌ 错误: 数据目录不存在: {args.data_dir}")
        return

    # 生成 DPO 数据
    generate_dpo_data(args)

    print("\n✅ DPO 数据生成完成！")
    print(f"\n下一步:")
    print(f"  python -u TALE-PT.py \\")
    print(f"      --strategy dpo \\")
    print(f"      --model_name Qwen-7B-Instruct \\")
    print(f"      --train_data_path {args.output_path} \\")
    print(f"      --output_dir ./results/tale_dpo \\")
    print(f"      --batch_size 2 \\")
    print(f"      --epoch 2 \\")
    print(f"      --lr 1e-5 \\")
    print(f"      --save")


if __name__ == "__main__":
    main()