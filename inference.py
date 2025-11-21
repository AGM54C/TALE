import os
import argparse
from utils import *
from llm_datasets import GSM8KZero, GSM8K, GPQA, MathBenchDataset, BanFakeNews
from llm_models import LLMModel
import time
import logging
from evaluator import AccEvaluator

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data(args):
    """
    根据命令行参数准备数据集

    Args:
        args: 包含数据集名称、推理模式、预算等配置的命令行参数

    Returns:
        相应类型的数据集对象，如果不支持则返回None
    """
    if 'math' in args.data_name:
        if args.data_name == 'math':
            dataset = MathBenchDataset(args, with_reasoning=args.reasoning, budget=args.budget, cache=False)
        else:
            dataset = MathBenchDataset(args, with_reasoning=args.reasoning, budget=args.budget,
                                       name=args.data_name, cache=False)

    elif args.data_name == 'GPQA':
        dataset = GPQA(args, with_reasoning=args.reasoning, budget=args.budget,
                       name=args.data_name, cache=False)
    elif args.data_name == 'GSM8K-Zero':
        dataset = GSM8KZero(args, with_reasoning=args.reasoning, budget=args.budget,
                            name=args.data_name, cache=False)
    elif args.data_name == 'GSM8K-Train':
        dataset = GSM8K(args, with_reasoning=args.reasoning, budget=args.budget,
                        name=args.data_name, cache=False, split='train')
    elif args.data_name == 'GSM8K-Test':
        dataset = GSM8K(args, with_reasoning=args.reasoning, budget=args.budget,
                        name=args.data_name, cache=False, split='test')
    elif args.data_name in ['BanFakeNews', 'BanFakeNews-Train', 'BanFakeNews-Test']:
        # 确定使用哪个子集
        if args.data_name == 'BanFakeNews-Train':
            subset_name = 'train'
        elif args.data_name == 'BanFakeNews-Test':
            subset_name = 'test'
        else:
            subset_name = 'all'

        dataset = BanFakeNews(args, with_reasoning=args.reasoning, budget=args.budget,
                              name=subset_name, cache=False)
    else:
        dataset = None
        ValueError(f"Not supported for {args.data_name}")
    return dataset


def data2list(dataset):
    """
    将数据集实例转换为样本列表和真值列表

    Args:
        dataset: 要转换的数据集

    Returns:
        tuple: (sample_list, gt_list)
            sample_list: 提示字符串列表
            gt_list: 真值答案列表
    """
    sample_list = []
    gt_list = []
    for idx, instance in enumerate(dataset):
        cur_sample = instance['round']
        ground_truth = instance['gold']
        sample_list.append(cur_sample[0]['prompt'])
        gt_list.append(ground_truth)
    return sample_list, gt_list


def inference_local(args, dataset, model, evaluator):
    """
    使用本地模型运行推理（例如Hugging Face模型）

    这个函数用于批量处理本地模型的推理任务，充分利用GPU进行并行计算。

    Args:
        args: 命令行参数
        dataset: 要运行推理的数据集
        model: 本地LLM模型实例
        evaluator: AccEvaluator实例用于计算准确率

    结果包含准确率百分比和每个样本的平均token消耗
    """
    acc_num = 0
    token_num = 0
    results = []
    start_time = time.time()
    logger.info("=" * 30 + 'Requesting' + "=" * 30 + '\n')
    # 将数据处理为列表格式
    logger.info(f"data size: {len(dataset)}")
    sample_list, gt_list = data2list(dataset)
    sample_list, gt_list = sample_list[args.start_index:args.end_index], gt_list[args.start_index:args.end_index]
    pred_list = model.query_batch(sample_list)
    # 输出到结果
    assert len(sample_list) == len(gt_list) == len(pred_list)
    for i in range(len(pred_list)):
        results.append({
            "ground truth": gt_list[i],
            "question": sample_list[i],
            "prediction": pred_list[i],
        })
        # 判断是否为分类任务（BanFakeNews）
        is_classification = 'BanFakeNews' in args.data_name
        acc_num += evaluator.evaluate_sample(results[-1],
                                             cloze=('cloze' in args.data_name) or (
                                                     args.data_name in ['GSM8K', 'GSM8K-Zero']),
                                             classification=is_classification)
        token_num += token_measure(pred_list[i])
    logger.info(f'Accuracy: {100 * acc_num / len(results):.2f}%')
    logger.info(f'Token costs: {token_num / len(results):.2f}')
    save_to_jsonl(results, args.output_path)
    logger.info(f'Time cost: {time.time() - start_time}')


def inference_api(args, dataset, model, evaluator, key):
    """
    使用API模型运行推理（例如GPT-4、Claude、Kimi-K2）

    这个函数逐个处理样本，适合API调用场景。它会记录每个样本的准确率和处理时间。

    Args:
        args: 命令行参数
        dataset: 要运行推理的数据集
        model: API模型实例
        evaluator: AccEvaluator实例用于计算准确率
        key: 模型访问的API密钥
    """
    acc_num = 0
    results = []
    start_time = time.time()
    logger.info("=" * 30 + 'Requesting' + "=" * 30 + '\n')
    if args.end_index is None:
        args.end_index = len(dataset)
    for idx, instance in enumerate(dataset):
        if args.start_index <= idx < args.end_index:
            cur_sample = instance['round']
            ground_truth = instance['gold']
            logger.info('=' * 30 + f"Step: {idx + 1} / {args.end_index}" + '=' * 30)
            logger.info(f"Question: {cur_sample[0]['prompt']}")
            pred = model.query(cur_sample, key=key)
            results.append({
                "ground truth": ground_truth,
                "question": cur_sample[0]['prompt'],
                "prediction": pred[0][0],
            })
            # 判断是否为分类任务（BanFakeNews）
            is_classification = 'BanFakeNews' in args.data_name
            acc_num += evaluator.evaluate_sample(results[-1],
                                                 cloze=('cloze' in args.data_name) or (
                                                         args.data_name in ['GSM8K', 'GSM8K-Zero']),
                                                 classification=is_classification)
            logger.info(f'Accuracy: {acc_num / len(results)}')
            save_to_jsonl(results, args.output_path)
            logger.info(f'Time cost: {time.time() - start_time}')


def parse_args():
    """
    解析推理脚本的命令行参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="=The budget token for our tech.")
    parser.add_argument("--budget", default=None, help="=The budget token for our tech.")
    parser.add_argument("--reasoning", action='store_true', help="If we use LLM reasoning.")
    parser.add_argument("--model", default='DeepSeek-R1-Distill-Qwen-1.5B', help="The model name on huggingface.")
    parser.add_argument("--output_path", default='./tmp',
                        help="The output path to save the model output.")
    parser.add_argument("--n", default=1, type=int, help="Number of samples from LLM.")
    parser.add_argument("--start_index", default=0, type=int, help="The start index for the dataset.")
    parser.add_argument("--end_index", default=None, type=int, help="The end index for the dataset.")
    parser.add_argument("--key_index", default=1, type=int, help="The key index for the dataset.")
    parser.add_argument("--data_name", default='GSM8K-Zero',
                        type=str, help="The dataset name used during our evaluation.")
    return parser.parse_args()


def main():
    """
    主函数：推理脚本的入口点
    """
    # 准备密钥和参数
    args = parse_args()
    args.output_path = os.path.join(args.output_path, args.model, args.data_name)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    args.output_path = os.path.join(args.output_path,
                                    'output_with_reasoning.jsonl'
                                    if args.reasoning else 'output_without_reasoning_new_prompt.jsonl')
    logger.info(f'Saving to {args.output_path}')

    # 判断是否为本地模型
    args.local = (args.model in ['Llama-3.1-8B-Instruct']) or 'Qwen' in args.model

    # 配置各个模型的API密钥
    # 这里为每个模型提供了两个key选项，可以通过key_index参数来选择
    keys = {
        'yi-lightning': ['your_api_key', 'your_api_key'],
        'gpt-4o-mini': ['your_api_key', 'your_api_key'],
        'gpt-4o-2024-05-13': ['your_api_key', 'your_api_key'],
        # 新增：Kimi-K2模型的API密钥配置
        # 支持多种名称格式以便灵活调用
        'kimi-k2': ['sk-bqcjhmsvgfwcdxaqlspqronplvnzfwcqctyrxubqqbredifa',
                    'sk-bqcjhmsvgfwcdxaqlspqronplvnzfwcqctyrxubqqbredifa'],
        'moonshotai/Kimi-K2-Thinking-Turbo': ['sk-bqcjhmsvgfwcdxaqlspqronplvnzfwcqctyrxubqqbredifa',
                                              'sk-bqcjhmsvgfwcdxaqlspqronplvnzfwcqctyrxubqqbredifa'],
    }
    key = keys[args.model][args.key_index] if not args.local else None

    # 准备数据集
    dataset = prepare_data(args)

    # 准备评估器
    evaluator = AccEvaluator(dataset)

    # 准备LLM模型
    model = LLMModel(args)
    if args.end_index is None:
        args.end_index = len(dataset)
    args.end_index = min(args.end_index, len(dataset))

    # 执行推理
    if args.local:
        inference_local(args, dataset, model, evaluator)
    else:
        inference_api(args, dataset, model, evaluator, key)


if __name__ == "__main__":
    main()