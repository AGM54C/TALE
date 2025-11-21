#!/usr/bin/env python3
import os
import random
import argparse
from utils import *
from torch.utils.data import Subset
from llm_datasets import *
from llm_models import LLMModel
import time
import logging
from evaluator import AccEvaluator

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """
    解析预算搜索脚本的命令行参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", default=None, help="=The budget token for our tech.")
    parser.add_argument("--do_search", action='store_true', help="If we search the best budget.")
    parser.add_argument("--model", default='gpt-4o-mini', help="yi-lightning, gpt-4o-mini")
    parser.add_argument("--output_path", default='tmp/search_budget/gpt-4o-mini',
                        help="The output path to save the model output.")
    parser.add_argument("--n", default=1, type=int, help="Number of samples from LLM.")
    parser.add_argument("--start_index", default=0, type=int, help="The start index for the dataset.")
    parser.add_argument("--end_index", default=700, type=int, help="The end index for the dataset.")
    parser.add_argument("--key_index", default=0, type=int, help="The key index for the dataset.")
    parser.add_argument("--data_name", default=None,
                        type=str, help="The dataset name used during our evaluation.")
    return parser.parse_args()


def search_budget(instance, budget, model, evaluator, key='your_api_key'):
    """
    搜索在保持预测准确性的同时最小化token的最优token预算

    这个函数实现了二分搜索策略来找到最优的token预算。它从一个初始预算开始，
    不断尝试减半的预算值，直到找到最小的能够保持答案正确的预算。

    工作原理：
    1. 从初始预算（通常是vanilla CoT的token数）开始
    2. 尝试使用budget/2的预算来回答问题
    3. 如果答案仍然正确且token使用减少，更新当前最优预算
    4. 继续减半直到答案不再正确或无法进一步减少

    Args:
        instance: 包含问题和真值的字典
        budget: 初始token预算（上界）
        model: LLM模型实例
        evaluator: AccEvaluator实例用于检查准确性
        key: 模型访问的API密钥（默认：'your_api_key'）

    Returns:
        tuple: (updated_instance, final_budget)
            updated_instance: 添加了预算信息的原始实例
            final_budget: 找到的最优预算
    """
    pred_flag = evaluator.evaluate_sample(instance)
    upper_bound = budget
    pre_token_cost = upper_bound
    instance['question_budget'] = 'None'
    instance['prediction_budget'] = 'None'
    instance['budget'] = upper_bound
    res_budget_list = [upper_bound]
    res_token_list = []

    while pred_flag:
        # 尝试减半的预算
        new_question = add_budget(instance['question'], budget // 2)
        cur_sample = [{'prompt': new_question}]
        cur_answer = model.query(cur_sample, key=key)[0][0]
        cur_token_cost = token_measure(cur_answer)
        res_token_list.append(cur_token_cost)

        # 检查答案是否仍然正确
        pred_flag = evaluator.evaluate_sample({
            'ground truth': instance['ground truth'],
            'prediction': cur_answer
        })

        # 下一次迭代的条件
        if pred_flag and cur_token_cost < pre_token_cost:
            # 更新当前最优答案和预算
            logger.info(f'Searching budget from {budget} to {budget // 2}.')
            logger.info(f'Token costs from {pre_token_cost} to {cur_token_cost}')
            instance['question_budget'] = new_question
            instance['prediction_budget'] = cur_answer
            instance['budget'] = budget // 2
            budget //= 2
            pre_token_cost = cur_token_cost
            res_budget_list.append(budget)
        else:
            break

    instance['budget_list'] = res_budget_list
    instance['token_list'] = res_token_list
    return instance, budget


def main():
    """
    主函数：预算搜索脚本的入口点

    这个脚本用于搜索每个问题的最优token预算。它对数据集中的每个样本
    执行二分搜索，找到能保持答案正确的最小token预算。
    """
    args = parse_args()
    args.do_search = True
    args.output_path = os.path.join(args.output_path, 'searched_budget.jsonl')
    logger.info(f'Saving to {args.output_path}')

    # 准备数据集
    if 'math' in args.data_name:
        if args.data_name == 'math':
            dataset = MathBenchDataset(args, with_reasoning=args.reasoning, budget=args.budget, cache=False)
        else:
            dataset = MathBenchDataset(args, with_reasoning=args.reasoning, budget=args.budget,
                                       name=args.data_name, cache=False)
    elif args.data_name == 'GSM8K-Zero':
        dataset = GSM8KZero(args, with_reasoning=args.reasoning, budget=args.budget,
                            name=args.data_name, cache=False)
    elif args.data_name == 'GSM8K':
        dataset = GSM8K(args, with_reasoning=args.reasoning, budget=args.budget,
                        name=args.data_name, cache=False)
    else:
        dataset = None
        ValueError(f"Not supported for {args.data_name}")

    model = LLMModel(args)

    # 配置各个模型的API密钥
    # 每个模型配置两个key选项，通过key_index参数选择
    keys = {
        'yi-lightning': ['your_api_key', 'your_api_key'],
        'gpt-4o-mini': ['your_api_key', 'your_api_key'],
        'gpt-4o-2024-05-13': ['your_api_key', 'your_api_key'],
        # 新增：Kimi-K2模型的API密钥配置
        'kimi-k2': ['sk-bqcjhmsvgfwcdxaqlspqronplvnzfwcqctyrxubqqbredifa',
                    'sk-bqcjhmsvgfwcdxaqlspqronplvnzfwcqctyrxubqqbredifa'],
        'moonshotai/Kimi-K2-Thinking-Turbo': ['sk-bqcjhmsvgfwcdxaqlspqronplvnzfwcqctyrxubqqbredifa',
                                              'sk-bqcjhmsvgfwcdxaqlspqronplvnzfwcqctyrxubqqbredifa'],
    }
    key = keys[args.model][args.key_index]

    res_budget = []
    logger.info("=" * 30 + 'Searching' + "=" * 30 + '\n')
    if args.end_index is None:
        args.end_index = len(dataset)

    if args.do_search:
        evaluator = AccEvaluator(dataset)
        for idx, instance in enumerate(dataset):
            if args.start_index <= idx < args.end_index:
                logger.info('=' * 30 + f"Step: {idx + 1} / {len(dataset)}" + '=' * 30)
                pred_flag = evaluator.evaluate_sample(instance)
                if not pred_flag:
                    # 跳过本来就答错的样本
                    continue
                target_pred = instance['prediction']
                budget_upper_bound = token_measure(target_pred)

                # 执行二分搜索找到最优预算
                new_instance, budget = search_budget(instance, budget_upper_bound,
                                                     model, evaluator, key=key)
                logger.info("Updating Budget: {}/{}.".format(budget, budget_upper_bound))
                logger.info("Updating Token costs: {}/{}."
                            .format(token_measure(new_instance['prediction_budget']), budget_upper_bound))
                res_budget.append(new_instance)
                save_to_jsonl(res_budget, args.output_path)
    else:
        # 不执行完整搜索，只是简单测试budget/2
        evaluator = AccEvaluator(dataset)
        for idx, instance in enumerate(dataset):
            if (idx + 1) >= 1:
                logger.info('=' * 30 + f"Step: {idx + 1} / {len(dataset)}" + '=' * 30)
                pred_flag = evaluator.evaluate_sample(instance)
                if not pred_flag:
                    continue
                target_pred = instance['prediction']
                budget_upper_bound = token_measure(target_pred)

                # 尝试使用budget/2
                new_question = add_budget(instance['question'], budget_upper_bound // 2)
                cur_sample = [{'prompt': new_question}]
                cur_answer = model.query(cur_sample, key=key)[0][0]
                cur_token_cost = token_measure(cur_answer)
                pred_flag = evaluator.evaluate_sample({
                    'ground truth': instance['ground truth'],
                    'prediction': cur_answer
                })
                if pred_flag:
                    instance['question_budget'] = new_question
                    instance['prediction_budget'] = cur_answer
                    instance['budget'] = budget_upper_bound // 2
                    logger.info("Updating Token costs: {}/{}."
                                .format(cur_token_cost, budget_upper_bound))
                    res_budget.append(instance)
                    save_to_jsonl(res_budget, args.output_path)


if __name__ == "__main__":
    main()