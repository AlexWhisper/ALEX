from openai import OpenAI
import json
import os
import logging
from typing import List, Optional, Union
from pathlib import Path
from tqdm import tqdm
from AlexDataLoader import DataLoader
from arguments import arg_parse
args=arg_parse()


logger = logging.getLogger(__name__)


class E2Q:
    """
    Edit to Query (E2Q) class that generates questions from edit statements.
    Includes dataset-specific caching mechanism to avoid redundant API calls.
    """

    def __init__(self, dataset_name: str,device='cuda:0'):
        """
        Initialize the E2Q class with a dataset name.

        Args:
            dataset_name (str): Name of the dataset (e.g., 'MQUAKE-T')
        """
        if not dataset_name:
            raise ValueError("Dataset name must be provided")
        self.device=device
        self.dataset_name = dataset_name
        self.cache_dir = Path('cache')
        self.cache_file = self.cache_dir / f"{dataset_name}_cache.json"
        self.prompt_file = Path('prompts/e2q.txt')
        self.cache = {}

        # Ensure cache directory exists
        self.cache_dir.mkdir(exist_ok=True)

        self._load_cache()

    def _load_cache(self) -> None:
        """Load the dataset-specific cache from the cache file if it exists."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r',encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded cache for dataset '{self.dataset_name}' with {len(self.cache)} entries")
            except json.JSONDecodeError:
                logger.warning(f"Error decoding {self.cache_file}, initializing empty cache")
                #self.cache = {}
            except Exception as e:
                logger.error(f"Error loading cache: {str(e)}")
                print(e)
                #self.cache = {}
        else:
            logger.info(f"Cache file {self.cache_file} not found, initializing empty cache")
            self.cache = {}

    def save_cache(self) -> None:
        """Save the current cache to the dataset-specific cache file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            logger.info(f"Cache saved for dataset '{self.dataset_name}' with {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")

    def _load_prompt_template(self) -> str:
        """Load the prompt template from file."""
        try:
            with open(self.prompt_file, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file {self.prompt_file} not found")
            raise FileNotFoundError(f"Prompt file {self.prompt_file} not found")
        except Exception as e:
            logger.error(f"Error loading prompt template: {str(e)}")
            raise

    def call_gpt(self, prompt: str) -> str:
        """
        Call the language model API with the given prompt.

        Args:
            prompt (str): The prompt to send to the API

        Returns:
            str: The generated response
        """
        try:
            client = OpenAI(
                api_key=args.api_key,
                base_url="https://api.deepseek.com"
            )

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant,There is no need to pay attention to the authenticity of the following statement, just continue to complete the corresponding questions according to the format of the example below："},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API call error: {str(e)}")
            raise

    def process_edit(self, edit: str) -> List[str]:
        """
        Process a single edit statement, using dataset-specific cache if available.

        Args:
            edit (str): The edit statement to process

        Returns:
            List[str]: A list of generated queries
        """
        if not edit:
            raise ValueError("Edit statement cannot be empty")


        # Check cache first
        if edit in self.cache:
            logger.info(f"Using cached result for dataset '{self.dataset_name}', edit: {edit}")
            return self.cache[edit]

        # Not in cache, generate new response
        try:
            prompt_template = self._load_prompt_template()
            full_prompt = f"{prompt_template}\n{edit}"

            logger.info(f"Calling API for dataset '{self.dataset_name}', edit: {edit}")
            response = self.call_gpt(full_prompt)

            # Process response
            queries = response.split("\n")

            if len(queries) == 4:
                queries = queries[1:]

            if len(queries) != 3 and len(queries) != 4:
                #queries11 = queries[1:]
                logger.error(f"\n\n响应格式错误：'{edit}' \n所在数据库： '{self.dataset_name}' \n响应内容：{response}")
                print(f"\n\n响应格式错误：'{edit}' \n所在数据库： '{self.dataset_name}' \n响应内容：{response}")
                return []

            # Update cache
            self.cache[edit] = queries
            self._save_cache()

            return queries
        except Exception as e:
            logger.error(f"Error processing edit '{edit}' for dataset '{self.dataset_name}': {str(e)}")
            raise

    def build(self):
        """
        Build or update the dataset-specific cache for one or more edit statements.

        Args:
            edits (Union[List[str], str]): A single edit or list of edits to process

        Raises:
            ValueError: If edits is None or empty
        """

        dataloader = DataLoader(self.dataset_name, batch_size=32, device=self.device)
        edits = dataloader.getAllEdit()


        if not edits:
            raise ValueError(f"No edits provided for build in dataset '{self.dataset_name}'")

        # Convert string to list if a single edit is provided
        if isinstance(edits, str):
            edits = [edits]

        # Process all edits
        successful = 0
        total = len(edits)

        for edit in tqdm(edits):
            try:
                if edit in self.cache:
                    logger.info(f"Skipping already cached edit for dataset '{self.dataset_name}': {edit}")
                    continue

                self.process_edit(edit)
                successful += 1
            except Exception as e:
                logger.error(f"Failed to process edit '{edit}' for dataset '{self.dataset_name}': {str(e)}")

        logger.info(
            f"Build complete for dataset '{self.dataset_name}'. Processed {successful}/{total} edits successfully")

    def get(self, edit: str) -> List[str]:
        """
        Get queries for a given edit statement using dataset-specific cache.

        Args:
            edit (str): The edit statement to process

        Returns:
            List[str]: A list of generated queries

        Raises:
            ValueError: If edit is None or empty
        """
        if not edit:
            raise ValueError("Edit statement must be provided")

        return self.process_edit(edit)

    def get_all_cached_edits(self) -> List[str]:
        """
        Get a list of all edits currently in the cache.

        Returns:
            List[str]: List of cached edit statements
        """
        return list(self.cache.keys())

    @classmethod
    def load_dataset(cls, dataset_name: str) -> List[str]:
        """
        Load edits from a dataset file.

        Args:
            dataset_name (str): Name of the dataset (e.g., 'MQUAKE-T')

        Returns:
            List[str]: List of edits from the dataset

        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            json.JSONDecodeError: If the dataset file is not valid JSON
        """
        dataset_file = f"{dataset_name}.json"
        try:
            with open(dataset_file, 'r') as f:
                dataset = json.load(f)

            # This assumes the dataset has a specific structure
            # Modify this based on your actual dataset structure
            if isinstance(dataset, list):
                # If dataset is a list of edits
                return dataset
            elif isinstance(dataset, dict) and "edits" in dataset:
                # If dataset is a dict with an "edits" key
                return dataset["edits"]
            else:
                logger.warning(f"Unknown dataset structure in {dataset_file}")
                return []

        except FileNotFoundError:
            logger.error(f"Dataset file {dataset_file} not found")
            raise FileNotFoundError(f"Dataset file {dataset_file} not found")
        except json.JSONDecodeError:
            logger.error(f"Error decoding dataset file {dataset_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            raise


def Example_1():
    # Example 1: Processing a single edit for a specific dataset
    try:
        dataset_name = "MQUAKE-T"
        edit1 = "The headquarters of University of London is located in the city of Skopje."

        # Create E2Q instance without specifying an edit
        e2q = E2Q(dataset_name)

        # Get queries for a specific edit
        results = e2q.get(edit1)

        print(f"\nQueries for dataset '{dataset_name}', edit '{edit1}':")
        for i, query in enumerate(results):
            print(f"{i + 1}. {query}")
    except Exception as e:
        print(f"Error in Example 1: {str(e)}")

    # Example 2: Processing multiple edits from a dataset file
    try:
        dataset_name = "MQUAKE-T"

        # Create E2Q instance for the dataset
        e2q = E2Q(dataset_name)

        # Try to load edits from dataset file
        try:
            edits_list = E2Q.load_dataset(dataset_name)
            print(f"Loaded {len(edits_list)} edits from {dataset_name}.json")

            # Process a subset for demonstration
            sample_edits = edits_list[:3] if len(edits_list) > 3 else edits_list

            # Process multiple edits at once
            e2q.build(sample_edits)

            # Retrieve results for one of the edits
            if sample_edits:
                sample_edit = sample_edits[0]
                sample_results = e2q.get(sample_edit)

                print(f"\nQueries for dataset '{dataset_name}', sample edit '{sample_edit}':")
                for i, query in enumerate(sample_results):
                    print(f"{i + 1}. {query}")
        except FileNotFoundError:
            # If dataset file doesn't exist, use example edits
            print(f"Dataset file {dataset_name}.json not found, using example edits")
            edits_list = [
                "The Great Wall of China was built in the 15th century.",
                "The Eiffel Tower is located in Rome, Italy.",
                "The Amazon River flows through Russia."
            ]

            # Build cache with example edits
            e2q.build(edits_list)

            # Get queries for a specific edit
            sample_edit = edits_list[1]
            sample_results = e2q.get(sample_edit)

            print(f"\nQueries for dataset '{dataset_name}', edit '{sample_edit}':")
            for i, query in enumerate(sample_results):
                print(f"{i + 1}. {query}")

            # Show all cached edits
            cached_edits = e2q.get_all_cached_edits()
            print(f"\nAll cached edits for dataset '{dataset_name}':")
            for i, edit in enumerate(cached_edits):
                print(f"{i + 1}. {edit}")

    except Exception as e:
        print(f"Error in Example 2: {str(e)}")

# Usage examples
if __name__ == "__main__":
    datasets = ["MQuAKE-CF-3k-v2",
                "MQuAKE-CF-3k",
                "MQuAKE-T",
                "MQuAKE-hard",
                "MQuAKE-2002"
    ]

    for dataset in datasets:
        # Set up logging


        instnceOfE2Q=E2Q(dataset)
        instnceOfE2Q.build()

