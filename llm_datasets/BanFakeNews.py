"""
BanFakeNews dataset module for loading and processing fake news detection data.
"""

import os
import pandas as pd
from torch.utils.data import Dataset
import logging
from utils import *

logger = logging.getLogger(__name__)


class BanFakeNews(Dataset):
    """
    Dataset class for BanFakeNews fake news detection.

    This dataset contains four CSV files:
    - Authentic-48K.csv: Authentic news articles
    - Fake-1K.csv: Fake news articles
    - LabeledAuthentic-7K.csv: Labeled authentic news
    - LabeledFake-1K.csv: Labeled fake news
    """

    def __init__(self, args, with_reasoning=True, name=None, cache=True, budget=None):
        """
        Initialize the BanFakeNews dataset.

        Args:
            args: Command line arguments containing configuration
            with_reasoning (bool): Whether to include step-by-step reasoning
            name (str, optional): Specific subset to load ('train', 'test', or None for all)
            cache (bool): Whether to cache the processed data
            budget (int, optional): Token budget for prompt generation
        """
        self.args = args
        self.cache = cache
        self.with_reasoning = with_reasoning
        self.name = name if name else 'all'

        # Update prompts with budget if provided
        if budget is not None:
            global banfakenews_prompts
            banfakenews_prompts = create_banfakenews_prompt(budget)

        self.banfakenews_std_data_sets = self._load_data()
        logger.info(f"Loading BanFakeNews dataset: {self.name}")
        self.dataset = sum(self.banfakenews_std_data_sets.values(), [])

    def _generate_configs(self):
        """
        Generate configuration for dataset loading.

        Returns:
            list: List of configuration dictionaries containing:
                - abbr: Dataset abbreviation
                - path: Path to dataset files
                - files: List of CSV files to load
                - reader_cfg: Input/output column configuration
                - meta_prompt: Prompt template configuration
        """
        config = [{
            'abbr': 'BanFakeNews',
            'path': './data/BanFakeNews',
            'files': {
                'train': ['Authentic-48K.csv', 'Fake-1K.csv'],
                'test': ['LabeledAuthentic-7K.csv', 'LabeledFake-1K.csv'],
                'all': ['Authentic-48K.csv', 'Fake-1K.csv',
                        'LabeledAuthentic-7K.csv', 'LabeledFake-1K.csv']
            },
            'reader_cfg': {
                'input_column': 'text',  # Assuming the text column name
                'output_column': 'label'
            },
            'meta_prompt': {
                'round': banfakenews_prompts['reasoning'] if self.with_reasoning else banfakenews_prompts[
                    'no_reasoning']
            }
        }]
        save_config(config[0])
        return config

    @staticmethod
    def _load_csv_file(filepath):
        """
        Load a single CSV file and return its contents.

        Args:
            filepath: Path to the CSV file

        Returns:
            pandas.DataFrame: Loaded data
        """
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            logger.info(f"Loaded {filepath}: {len(df)} samples")
            return df
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            try:
                df = pd.read_csv(filepath, encoding='latin-1')
                logger.info(f"Loaded {filepath} with latin-1 encoding: {len(df)} samples")
                return df
            except Exception as e2:
                logger.error(f"Failed to load {filepath}: {e2}")
                return pd.DataFrame()

    @staticmethod
    def _generate_std_subset(raw_data, cfg):
        """
        Generate standardized subset of the dataset.

        Args:
            raw_data: Raw data from the dataset
            cfg: Configuration dictionary

        Returns:
            list: List of processed examples, each containing:
                - gold: Ground truth label ('Real' or 'Fake')
                - round: List of conversation turns with human and bot messages
        """
        examples = []
        prompt_template = cfg["meta_prompt"]["round"][0]['prompt']

        for item in raw_data:
            examples.append({
                'gold': item['label'],
                'round': [
                    {
                        "role": "HUMAN",
                        "prompt": prompt_template.replace("{question}", item['text'])
                    },
                    {
                        "role": "BOT",
                        "prompt": "{answer}"
                    }
                ]
            })
        return examples

    def _generate_formal_info(self, cfg):
        """
        Generate formalized information from raw CSV files.

        Args:
            cfg: Configuration dictionary

        Returns:
            list: List of processed data items, each containing:
                - text: The news article text
                - label: 'Real' or 'Fake'
        """
        data = []
        files_to_load = cfg['files'].get(self.name, cfg['files']['all'])

        for filename in files_to_load:
            filepath = os.path.join(cfg['path'], filename)

            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                continue

            df = self._load_csv_file(filepath)

            if df.empty:
                continue

            # Determine label based on filename
            if 'Authentic' in filename or 'authentic' in filename:
                label = 'Real'
            elif 'Fake' in filename or 'fake' in filename:
                label = 'Fake'
            else:
                label = 'Unknown'

            # Process each row
            # Adjust column names based on actual CSV structure
            # Common column names: 'text', 'content', 'article', 'news', etc.
            possible_text_columns = ['text', 'content', 'article', 'news', 'headline',
                                     'title', 'body', 'full_text', 'Text', 'Content']

            text_column = None
            for col in possible_text_columns:
                if col in df.columns:
                    text_column = col
                    break

            if text_column is None:
                # If no known column found, use the first column
                text_column = df.columns[0]
                logger.info(f"Using first column '{text_column}' as text column")

            for idx, row in df.iterrows():
                text = str(row[text_column]).strip()

                # Skip empty or very short texts
                if len(text) < 10:
                    continue

                data.append({
                    'text': text,
                    'label': label
                })

        logger.info(f"Total samples loaded: {len(data)}")
        return data

    def _load_data(self):
        """
        Load and process the dataset.

        Returns:
            dict: Dictionary mapping dataset abbreviations to processed subsets
        """
        cfgs = self._generate_configs()
        save_config(cfgs[0])
        std_data_sets = {}

        # Create cache directory if needed
        cache_dir = './.cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            logger.info(f"Created cache directory: {cache_dir}")

        for cfg in cfgs:
            cache_file = os.path.join(cache_dir, f"{cfg['abbr']}_{self.name}.jsonl")

            # Try to load from cache
            if os.path.exists(cache_file) and self.cache:
                logger.info(f"Loading from cache: {cache_file}")
                std_subset = read_jsonl(cache_file)
            else:
                # Load and process data
                info = self._generate_formal_info(cfg)
                std_subset = self._generate_std_subset(info, cfg)
                save_to_jsonl(std_subset, cache_file)

            std_data_sets[cfg["abbr"]] = std_subset

        return std_data_sets

    def __len__(self):
        """
        Get the total number of examples in the dataset.

        Returns:
            int: Number of examples
        """
        return len(self.dataset)

    def __getitem__(self, i):
        """
        Get a specific example from the dataset.

        Args:
            i: Index of the example to retrieve

        Returns:
            dict: The example at index i
        """
        return self.dataset[i]


# Prompt templates for BanFakeNews
banfakenews_prompts = {
    'reasoning': [
        dict(role='HUMAN', prompt="""Q: {question}

Please determine if the above news article is Real or Fake. Analyze the content carefully and provide your reasoning.

Please give the response by strictly following this format: [[label]], for example: Label: [[Real]] or Label: [[Fake]].

Let's think step by step:
"""),
        dict(role='BOT', prompt='A: {answer}'),
    ],
    'no_reasoning': [
        dict(role='HUMAN', prompt="""Please determine if the following news article is Real or Fake. Give the answer directly without reasoning.

Please give the response by strictly following this format: [[label]], for example: Label: [[Real]] or Label: [[Fake]].

Q: {question}
"""),
        dict(role='BOT', prompt='A: {answer}'),
    ],
}


def create_banfakenews_prompt(budget=512):
    """
    Create BanFakeNews prompts with token budget constraints.

    Args:
        budget: Token limit for response (default: 512)

    Returns:
        dict: Modified banfakenews_prompts with budget constraints added
    """
    new = banfakenews_prompts['reasoning'][0]['prompt'] \
        .replace("Let's think step by step:\n", f"Let's think step by step and use less than {budget} tokens:\n")
    banfakenews_prompts['reasoning'][0]['prompt'] = new

    return banfakenews_prompts