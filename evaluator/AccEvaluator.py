import logging
import re
from math_verify import parse, verify

logger = logging.getLogger(__name__)


class AccEvaluator:
    """
    A class for evaluating the accuracy of predictions against ground truth in various formats.
    """

    def __init__(self, dataset=None):
        """
        Initialize the AccEvaluator with an optional dataset.

        Args:
            dataset: Optional dataset to evaluate. If None, must be set later.
        """
        self.dataset = dataset

    def accuracy(self):
        """
        Calculate the overall accuracy across the entire dataset.

        Returns:
            float: The accuracy score as a ratio of correct predictions to total samples
        """
        acc_num = 0
        for sample in self.dataset:
            acc_num += self.evaluate_sample(sample)
        return acc_num / len(self.dataset)

    @staticmethod
    def find_answer(text):
        """
        Extract multiple choice answer (A, B, C, or D) from text response.

        Args:
            text: The text response to analyze

        Returns:
            str: The extracted answer choice (A, B, C, D) or 'None' if not found

        """
        text = text.strip()
        last_newline_index = text.rfind('\n')
        prediction = text[last_newline_index + 1:]
        if len(prediction) < 5:
            search_texts = [
                'the correct answer is',
                '答案为选项'
            ]
            for search_text in search_texts:
                index = text.find(search_text)
                if index != -1:
                    prediction = text[index:]
                    break

        pattern = re.compile(r'[ABCD]')
        matches = pattern.findall(prediction)
        if matches:

            answer = ''.join(matches)[-1]
        else:
            answer = 'None'

        return answer

    @staticmethod
    def extract_predicted_answer(text):
        """
        Extract numerical or text answer from a response.

        Args:
            text: The text response to analyze

        Returns:
            str or None: The extracted answer or None if no valid answer found

        """
        pattern = r"\[\[(.*?)\]\]"

        match = re.findall(pattern, text)

        if match:
            return match[-1]

        regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
        regexes_to_ignore = [
            ",",
            "\\$",
            "(?s).*#### ",
            "\\.$"
        ]
        match = re.findall(regex_pattern, text)
        if match:
            match = match[-1]
            if isinstance(match, tuple):
                match = [m for m in match if m][0]
            text = match.strip()
            for regex in regexes_to_ignore:
                text = re.sub(regex, "", text)
            return text
        else:
            return None

    @staticmethod
    def extract_classification_label(text):
        """
        Extract classification label (Real/Fake, True/False, etc.) from text response.

        Args:
            text: The text response to analyze

        Returns:
            str or None: The extracted label or None if not found
        """
        # 首先尝试从 [[label]] 格式中提取
        pattern = r"\[\[(.*?)\]\]"
        match = re.findall(pattern, text)
        if match:
            label = match[-1].strip()
            # 标准化标签
            label_lower = label.lower()
            if 'real' in label_lower or 'authentic' in label_lower or 'true' in label_lower:
                return 'Real'
            elif 'fake' in label_lower or 'false' in label_lower:
                return 'Fake'
            return label

        # 如果没有找到 [[]] 格式，尝试其他模式
        text_lower = text.lower()

        # 查找常见的分类关键词
        if 'label:' in text_lower or 'answer:' in text_lower:
            # 提取标签后的内容
            for keyword in ['label:', 'answer:']:
                if keyword in text_lower:
                    idx = text_lower.index(keyword) + len(keyword)
                    remaining = text[idx:].strip()
                    # 提取前几个词
                    words = remaining.split()[:3]
                    for word in words:
                        word_clean = word.strip('.,!?;:').lower()
                        if 'real' in word_clean or 'authentic' in word_clean or 'true' in word_clean:
                            return 'Real'
                        elif 'fake' in word_clean or 'false' in word_clean:
                            return 'Fake'

        # 最后，在整个文本中搜索关键词
        if 'real' in text_lower or 'authentic' in text_lower:
            return 'Real'
        elif 'fake' in text_lower:
            return 'Fake'

        return None

    def evaluate_sample(self, sample, cloze=True, classification=False):
        """
        Evaluate a single sample against its ground truth.

        Args:
            sample: Dictionary containing 'ground truth' and 'prediction' keys
            cloze: Boolean indicating if this is a cloze-style question (True) or
                  multiple choice (False)
            classification: Boolean indicating if this is a classification task

        Returns:
            bool: True if the prediction matches ground truth, False otherwise

        """
        gt = sample['ground truth']
        pred = sample['prediction']

        # 处理分类任务（如 BanFakeNews）
        if classification:
            extracted_label = self.extract_classification_label(pred)
            if extracted_label is None:
                logger.warning(f"Could not extract label from prediction: {pred[:100]}")
                return False

            # 标准化比较
            gt_normalized = gt.strip().lower()
            pred_normalized = extracted_label.strip().lower()

            # 支持多种标签格式
            match = (gt_normalized == pred_normalized) or \
                    (gt_normalized in pred_normalized) or \
                    (pred_normalized in gt_normalized)

            if match:
                return True

            # 尝试同义词匹配
            real_synonyms = ['real', 'authentic', 'true', 'genuine']
            fake_synonyms = ['fake', 'false', 'fabricated', 'misinformation']

            gt_is_real = any(syn in gt_normalized for syn in real_synonyms)
            gt_is_fake = any(syn in gt_normalized for syn in fake_synonyms)
            pred_is_real = any(syn in pred_normalized for syn in real_synonyms)
            pred_is_fake = any(syn in pred_normalized for syn in fake_synonyms)

            return (gt_is_real and pred_is_real) or (gt_is_fake and pred_is_fake)

        # 处理填空题或数学题
        if cloze:
            return (gt == self.extract_predicted_answer(pred)) or (f"[[{gt}]]" in pred) \
                or verify(parse(gt), parse(self.extract_predicted_answer(pred)))
        # 处理选择题
        else:
            if f'[[{gt}]]' in pred:
                return True
            choice = self.find_answer(sample['prediction'])
            return choice == gt