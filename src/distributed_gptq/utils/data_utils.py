"""
Data loading utilities for distributed GPTQ quantization.

This module provides utilities for loading and preparing calibration data
for the quantization process.
"""

import torch
import torch.utils.data as data
from typing import List, Union, Dict, Any, Optional, Callable, Iterator
import json
import random
import numpy as np
from pathlib import Path


class CalibrationDataset(data.Dataset):
    """
    Dataset wrapper for calibration data.
    """
    
    def __init__(
        self,
        data_source: Union[List, str, Path],
        tokenizer: Optional[Callable] = None,
        max_length: int = 512,
        preprocessing_fn: Optional[Callable] = None
    ):
        """
        Initialize calibration dataset.
        
        Args:
            data_source: List of samples, file path, or data source
            tokenizer: Tokenizer function
            max_length: Maximum sequence length
            preprocessing_fn: Custom preprocessing function
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessing_fn = preprocessing_fn
        
        # Load data
        if isinstance(data_source, (str, Path)):
            self.data = self._load_from_file(data_source)
        elif isinstance(data_source, list):
            self.data = data_source
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")
    
    def _load_from_file(self, file_path: Union[str, Path]) -> List[Any]:
        """
        Load data from file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            List of data samples
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_path.suffix == '.jsonl':
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        elif file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Tokenized sample
        """
        sample = self.data[idx]
        
        # Apply preprocessing if provided
        if self.preprocessing_fn:
            sample = self.preprocessing_fn(sample)
        
        # Extract text from sample
        if isinstance(sample, dict):
            text = sample.get('text', sample.get('content', str(sample)))
        else:
            text = str(sample)
        
        # Tokenize if tokenizer is provided
        if self.tokenizer:
            return self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # Return raw text
        return {'text': text}


class DataCollator:
    """
    Collate function for batching calibration data.
    """
    
    def __init__(self, pad_token_id: int = 0, return_tensors: str = 'pt'):
        """
        Initialize data collator.
        
        Args:
            pad_token_id: Padding token ID
            return_tensors: Return tensor format
        """
        self.pad_token_id = pad_token_id
        self.return_tensors = return_tensors
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.
        
        Args:
            batch: List of samples
            
        Returns:
            Batched tensors
        """
        if not batch:
            return {}
        
        # Handle single sample
        if len(batch) == 1:
            return {k: v.squeeze(0) if v.dim() > 1 else v for k, v in batch[0].items()}
        
        # Batch multiple samples
        batch_dict = {}
        for key in batch[0].keys():
            if key == 'text':
                batch_dict[key] = [sample[key] for sample in batch]
            else:
                # Stack tensors
                tensors = [sample[key].squeeze(0) if sample[key].dim() > 1 else sample[key] for sample in batch]
                if tensors[0].dim() == 0:
                    # Scalar values
                    batch_dict[key] = torch.stack(tensors)
                else:
                    # Sequence data - pad to max length
                    max_len = max(t.size(-1) for t in tensors)
                    padded_tensors = []
                    
                    for tensor in tensors:
                        if tensor.size(-1) < max_len:
                            pad_size = max_len - tensor.size(-1)
                            if tensor.dim() == 1:
                                padded = torch.cat([tensor, torch.full((pad_size,), self.pad_token_id)])
                            else:
                                pad_shape = list(tensor.shape)
                                pad_shape[-1] = pad_size
                                padded = torch.cat([tensor, torch.full(pad_shape, self.pad_token_id)], dim=-1)
                        else:
                            padded = tensor
                        padded_tensors.append(padded)
                    
                    batch_dict[key] = torch.stack(padded_tensors)
        
        return batch_dict


def prepare_calibration_data(
    data: Union[List, str, Path],
    tokenizer: Optional[Callable] = None,
    max_length: int = 512,
    num_samples: Optional[int] = None,
    shuffle: bool = False,
    preprocessing_fn: Optional[Callable] = None,
    batch_size: int = 1,
    **dataloader_kwargs,
) -> data.DataLoader:
    """Create a simple calibration dataloader from raw data."""

    return create_calibration_dataloader(
        data_source=data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_samples=num_samples,
        shuffle=shuffle,
        preprocessing_fn=preprocessing_fn,
        **dataloader_kwargs,
    )


def create_calibration_dataloader(
    data_source: Union[List, str, Path],
    tokenizer: Optional[Callable] = None,
    batch_size: int = 1,
    max_length: int = 512,
    num_samples: Optional[int] = None,
    shuffle: bool = False,
    preprocessing_fn: Optional[Callable] = None,
    **dataloader_kwargs
) -> data.DataLoader:
    """
    Create a DataLoader for calibration data.
    
    Args:
        data_source: Data source (list, file path, etc.)
        tokenizer: Tokenizer function
        batch_size: Batch size
        max_length: Maximum sequence length
        num_samples: Number of samples to use (None for all)
        shuffle: Whether to shuffle data
        preprocessing_fn: Custom preprocessing function
        **dataloader_kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader for calibration
    """
    # Create dataset
    dataset = CalibrationDataset(
        data_source=data_source,
        tokenizer=tokenizer,
        max_length=max_length,
        preprocessing_fn=preprocessing_fn
    )
    
    # Subsample if requested
    if num_samples and num_samples < len(dataset):
        indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(indices)
        indices = indices[:num_samples]
        dataset = data.Subset(dataset, indices)
    
    # Create collator
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0) if tokenizer else 0
    collator = DataCollator(pad_token_id=pad_token_id)
    
    # Create DataLoader
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and num_samples is None,
        collate_fn=collator,
        **dataloader_kwargs
    )


def create_dataloader(
    data: Union[List, str, Path],
    tokenizer: Optional[Callable] = None,
    batch_size: int = 1,
    max_length: int = 512,
    num_samples: Optional[int] = None,
    shuffle: bool = False,
    preprocessing_fn: Optional[Callable] = None,
    **dataloader_kwargs,
) -> data.DataLoader:
    """Alias for :func:`create_calibration_dataloader` for backward compatibility."""

    return create_calibration_dataloader(
        data_source=data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_samples=num_samples,
        shuffle=shuffle,
        preprocessing_fn=preprocessing_fn,
        **dataloader_kwargs,
    )


def load_common_datasets(dataset_name: str, split: str = 'train', num_samples: int = 1000) -> List[str]:
    """
    Load common calibration datasets.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split to use
        num_samples: Number of samples to load
        
    Returns:
        List of text samples
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required for loading common datasets")
    
    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 50]
    elif dataset_name == 'c4':
        dataset = load_dataset('c4', 'en', split=split, streaming=True)
        texts = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            if len(item['text'].strip()) > 100:
                texts.append(item['text'])
    elif dataset_name == 'ptb':
        dataset = load_dataset('ptb_text_only', split=split)
        texts = [item['sentence'] for item in dataset]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Sample and return
    if len(texts) > num_samples:
        texts = random.sample(texts, num_samples)
    
    return texts


class DistributedSampler(data.DistributedSampler):
    """
    Distributed sampler for calibration data.
    """
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        """
        Initialize distributed sampler.
        
        Args:
            dataset: Dataset to sample from
            num_replicas: Number of processes participating in distributed training
            rank: Rank of the current process
            shuffle: Whether to shuffle the data
            seed: Random seed
        """
        super().__init__(dataset, num_replicas, rank, shuffle, seed)
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over dataset indices."""
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # Add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        
        # Subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        return iter(indices)


def create_distributed_dataloader(
    data_source: Union[List, str, Path],
    tokenizer: Optional[Callable] = None,
    batch_size: int = 1,
    max_length: int = 512,
    num_samples: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
    **kwargs
) -> data.DataLoader:
    """
    Create a distributed DataLoader for calibration data.
    
    Args:
        data_source: Data source
        tokenizer: Tokenizer function
        batch_size: Batch size per process
        max_length: Maximum sequence length
        num_samples: Total number of samples
        rank: Process rank
        world_size: Total number of processes
        **kwargs: Additional arguments
        
    Returns:
        Distributed DataLoader
    """
    # Create dataset
    dataset = CalibrationDataset(
        data_source=data_source,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create collator
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0) if tokenizer else 0
    collator = DataCollator(pad_token_id=pad_token_id)
    
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        **kwargs
    )
