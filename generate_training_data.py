#!/usr/bin/env python3
"""
生成 BanFakeNews 数据集的 TALE-PT-SFT 训练数据

这个脚本将:
1. 加载 BanFakeNews 数据集
2. 为每个样本生成带预算约束的 prompt
3. 生成相应的 completion (模拟训练数据)
4. 保存为 JSONL 格式供 TALE-PT 训练使用
"""

import os
import json
import argparse
import random
from tqdm import tqdm


def generate_sft_prompt(text, budget=128):
    """
    生成 SFT 训练的 prompt

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


def generate_completion(label, reasoning_style="simple"):
    """
    生成训练用的 completion 示例

    Args:
        label: 真实标签 ('Real' 或 'Fake')
        reasoning_style: 推理风格 ('simple', 'detailed', 'critical')

    Returns:
        completion 字符串
    """
    if label == "Real":
        if reasoning_style == "simple":
            completions = [
                "After analyzing the content, the article presents factual information with credible sources and proper journalistic standards. The language is neutral and the claims are verifiable. Label: [[Real]]",
                "This article appears authentic based on: 1) Credible sources cited, 2) Consistent facts, 3) Professional writing style, 4) Verifiable claims. Label: [[Real]]",
                "The content shows characteristics of genuine journalism: proper sourcing, balanced perspective, and factual accuracy. No signs of manipulation or misinformation detected. Label: [[Real]]"
            ]
        elif reasoning_style == "detailed":
            completions = [
                "Examining this article carefully: The sources mentioned are reputable and can be verified. The writing maintains objectivity without emotional manipulation. Facts presented align with known information. The author demonstrates expertise in the subject. No red flags for misinformation detected. Label: [[Real]]",
                "Analysis reveals: (1) Multiple credible sources cited, (2) Claims are specific and verifiable, (3) Language is professional and measured, (4) No sensationalism or emotional manipulation, (5) Consistent with established facts. Conclusion: This is authentic reporting. Label: [[Real]]"
            ]
        else:  # critical
            completions = [
                "Critical evaluation: Sources are verifiable and reputable. Cross-referencing with known databases confirms the information. The narrative structure follows standard journalistic practices. No logical inconsistencies or false claims detected. Label: [[Real]]"
            ]
    else:  # Fake
        if reasoning_style == "simple":
            completions = [
                "This article exhibits several red flags: unverified claims, sensational language, lack of credible sources, and factual inconsistencies. These are typical characteristics of misinformation. Label: [[Fake]]",
                "Key indicators of fake news present: 1) No credible sources, 2) Sensational headlines, 3) Unverifiable claims, 4) Emotional manipulation. Label: [[Fake]]",
                "The content shows signs of fabrication: lack of proper sourcing, inconsistent facts, biased language, and unverifiable claims. Label: [[Fake]]"
            ]
        elif reasoning_style == "detailed":
            completions = [
                "Detailed analysis reveals multiple issues: (1) Sources are either absent or unverifiable, (2) Claims contradict established facts, (3) Language is highly emotional and manipulative, (4) Logical inconsistencies throughout, (5) Designed to provoke rather than inform. This is misinformation. Label: [[Fake]]",
                "Red flags identified: The article lacks credible attribution, makes extraordinary claims without evidence, uses sensational language, contains factual errors, and appears designed to mislead. These are hallmarks of fake news. Label: [[Fake]]"
            ]
        else:  # critical
            completions = [
                "Critical assessment: Cross-checking reveals factual inaccuracies. Sources cannot be verified. The narrative employs emotional manipulation tactics. Claims are inconsistent with established information. This is fabricated content. Label: [[Fake]]"
            ]

    return random.choice(completions)


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


def generate_training_data(args):
    """
    生成训练数据
    """
    print("=" * 60)
    print("生成 BanFakeNews TALE-PT-SFT 训练数据")
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

    # 限制数据量（如果指定）
    if args.max_samples and args.max_samples < len(train_data):
        train_data = train_data[:args.max_samples]
        print(f"✓ 限制为前 {args.max_samples} 条样本")

    print(f"\n总训练样本: {len(train_data)} 条")

    # 生成训练数据
    print("\n生成 SFT 训练数据...")
    sft_data = []

    for item in tqdm(train_data, desc="生成训练数据"):
        # 随机选择预算
        budget = random.choice(args.budgets)

        # 生成 prompt
        prompt = generate_sft_prompt(item['text'], budget)

        # 生成 completion
        reasoning_style = random.choice(['simple', 'detailed', 'critical'])
        completion = generate_completion(item['label'], reasoning_style)

        sft_data.append({
            'prompt': prompt,
            'completion': completion,
            'label': item['label'],
            'budget': budget
        })

    # 保存训练数据
    output_path = args.output_path
    print(f"\n保存训练数据到: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sft_data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"✓ 已保存 {len(sft_data)} 条训练数据")

    # 保存统计信息
    stats = {
        'total_samples': len(sft_data),
        'real_samples': sum(1 for x in sft_data if x['label'] == 'Real'),
        'fake_samples': sum(1 for x in sft_data if x['label'] == 'Fake'),
        'budgets': args.budgets,
        'avg_text_length': sum(len(x['prompt']) for x in sft_data) / len(sft_data)
    }

    print("\n" + "=" * 60)
    print("统计信息:")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("=" * 60)


def generate_test_data(args):
    """
    生成测试数据
    """
    print("=" * 60)
    print("生成 BanFakeNews 测试数据")
    print("=" * 60)

    # 加载测试集数据
    print("\n加载测试集数据...")
    test_data = []

    # 标注的真实新闻
    labeled_authentic_path = os.path.join(args.data_dir, 'LabeledAuthentic-7K.csv')
    if os.path.exists(labeled_authentic_path):
        authentic_data = load_csv_data(labeled_authentic_path, 'Real')
        print(f"✓ 加载标注真实新闻: {len(authentic_data)} 条")
        test_data.extend(authentic_data)

    # 标注的假新闻
    labeled_fake_path = os.path.join(args.data_dir, 'LabeledFake-1K.csv')
    if os.path.exists(labeled_fake_path):
        fake_data = load_csv_data(labeled_fake_path, 'Fake')
        print(f"✓ 加载标注假新闻: {len(fake_data)} 条")
        test_data.extend(fake_data)

    if len(test_data) == 0:
        print("❌ 错误: 没有找到测试数据！")
        return

    # 打乱数据
    random.shuffle(test_data)

    # 限制数据量
    if args.max_test_samples and args.max_test_samples < len(test_data):
        test_data = test_data[:args.max_test_samples]
        print(f"✓ 限制为前 {args.max_test_samples} 条样本")

    print(f"\n总测试样本: {len(test_data)} 条")

    # 生成测试数据（使用默认预算）
    print("\n生成测试数据...")
    formatted_test_data = []

    default_budget = 128
    for item in tqdm(test_data, desc="生成测试数据"):
        question = generate_sft_prompt(item['text'], default_budget)

        formatted_test_data.append({
            'question': question,
            'ground_truth': item['label']
        })

    # 保存测试数据
    output_path = args.test_output_path
    print(f"\n保存测试数据到: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in formatted_test_data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"✓ 已保存 {len(formatted_test_data)} 条测试数据")


def parse_args():
    parser = argparse.ArgumentParser(description="生成 BanFakeNews TALE-PT 训练数据")

    # 数据路径
    parser.add_argument("--data_dir", default="./data/BanFakeNews",
                        help="BanFakeNews CSV 文件目录")
    parser.add_argument("--output_path", default="./data/banfakenews_train_sft.jsonl",
                        help="训练数据输出路径")
    parser.add_argument("--test_output_path", default="./data/banfakenews_test.jsonl",
                        help="测试数据输出路径")

    # 数据生成选项
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大训练样本数（None 表示使用全部）")
    parser.add_argument("--max_test_samples", type=int, default=1000,
                        help="最大测试样本数")
    parser.add_argument("--budgets", type=int, nargs='+', default=[64, 128, 256],
                        help="token 预算选项列表")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    # 生成选项
    parser.add_argument("--generate_train", action='store_true',
                        help="生成训练数据")
    parser.add_argument("--generate_test", action='store_true',
                        help="生成测试数据")

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子
    random.seed(args.seed)

    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"❌ 错误: 数据目录不存在: {args.data_dir}")
        return

    # 生成训练数据
    if args.generate_train:
        generate_training_data(args)

    # 生成测试数据
    if args.generate_test:
        generate_test_data(args)

    # 如果都没指定，默认生成两者
    if not args.generate_train and not args.generate_test:
        generate_training_data(args)
        print("\n")
        generate_test_data(args)

    print("\n✅ 数据生成完成！")


if __name__ == "__main__":
    main()