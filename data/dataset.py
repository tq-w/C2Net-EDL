"""
Dataset classes for ODIR multi-label classification with long-tail handling.
Adapted for the new multi-label CSV format from FINAL_CORRECTED_VERSION.
"""

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ODIRDatasetEDL(Dataset):
    """
    ODIR (Ophthalmic Disease Intelligent Recognition) dataset with EDL support.
    Handles the new multi-label format where labels are ';' separated indices.
    """

    def __init__(
        self,
        data_frame,
        root_dir,
        mode='train',
        transform=None,
        enhance_dir=None,
        grade_csv=None,
        split_column='random_mode',
        unusable_column='unusable',
        min_samples=10,
        class_map=None
    ):
        # Filter by mode
        mask = (data_frame[split_column] == mode)
        # Check if unusable column exists before filtering
        if unusable_column in data_frame.columns:
            mask &= (data_frame[unusable_column] == 0)
            
        self.data_frame = data_frame[mask].copy()
        self.root_dir = root_dir
        self.transform = transform
        self.enhance_dir = enhance_dir
        self.mode = mode
        self.min_samples = min_samples

        # Load grade CSV for quality-based image selection
        if grade_csv is not None and os.path.exists(grade_csv):
            self.grade_csv = pd.read_csv(grade_csv)
        else:
            self.grade_csv = None

        # Determine total number of classes (20 fine-grained classes per paper)
        self.num_classes = 20
        
        # In this new version, we don't necessarily need min_samples filtering 
        # for class removal because we use fixed 30 classes, 
        # but we'll keep the structure for compatibility.
        self.valid_classes = list(range(self.num_classes))
        self.class_map = {i: i for i in range(self.num_classes)}

        # Apply class filtering to labels
        self._filter_labels()

    def _filter_labels(self):
        """
        Convert ';' separated label strings to multi-hot vectors.
        """
        filtered_labels = []
        for idx in range(len(self.data_frame)):
            label_str = str(self.data_frame.iloc[idx]['labels'])
            multi_hot = np.zeros(self.num_classes, dtype=np.float32)
            
            # Parse labels separated by ';'
            if label_str and label_str != 'nan':
                indices = [int(i) for i in label_str.split(';') if i.strip()]
                for i in indices:
                    if 0 <= i < self.num_classes:
                        multi_hot[i] = 1.0
            
            filtered_labels.append(multi_hot)

        self.filtered_labels = np.array(filtered_labels, dtype=np.float32)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        """
        # Determine image directory based on quality
        image_name = self.data_frame.iloc[idx]['image_name']
        
        if self.enhance_dir and self.grade_csv is not None:
            try:
                grade = self.grade_csv.loc[
                    self.grade_csv['image_name'] == image_name, 'grade'
                ].values[0]
                img_dir = self.enhance_dir if grade != 'good' else self.root_dir
            except (IndexError, KeyError):
                img_dir = self.root_dir
        else:
            img_dir = self.root_dir

        # Handle extension: CSV has .jpg, files are .png
        base_name = os.path.splitext(image_name)[0]
        img_path = os.path.join(img_dir, base_name + '.png')
        
        # Fallback if .png not found, try original name from CSV
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, image_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (256, 256), (0, 0, 0))

        # Get filtered labels
        labels = torch.tensor(self.filtered_labels[idx]).float()

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, labels

    def get_class_map(self):
        """Return the class mapping dictionary."""
        return self.class_map.copy()

    def get_num_classes(self):
        """Return the number of valid classes."""
        return self.num_classes


def split_dataset(dataframe, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0):
    """
    Split dataset into train/val/test sets and add split column.
    """
    np.random.seed(seed)
    num_samples = len(dataframe)
    indices = np.arange(num_samples)

    # Create split
    num_train = int(train_ratio * num_samples)
    num_val = int(val_ratio * num_samples)

    np.random.shuffle(indices)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    # Add split column
    dataframe['random_mode'] = None
    dataframe.loc[dataframe.index[train_indices], 'random_mode'] = 'train'
    dataframe.loc[dataframe.index[val_indices], 'random_mode'] = 'val'
    dataframe.loc[dataframe.index[test_indices], 'random_mode'] = 'test'

    return dataframe


def analyze_dataset_distribution(csv_path, min_samples=10):
    """
    Analyze the dataset distribution for the new multi-label format.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # Ensure split column exists
    if 'random_mode' not in df.columns:
        df = split_dataset(df, seed=0)

    # Get training data
    train_df = df[df['random_mode'] == 'train']

    num_classes = 20
    class_counts = np.zeros(num_classes)
    
    for label_str in train_df['labels']:
        if pd.isna(label_str): continue
        indices = [int(i) for i in str(label_str).split(';') if i.strip()]
        for i in indices:
            if 0 <= i < num_classes:
                class_counts[i] += 1

    print("Dataset Distribution Analysis:")
    print("=" * 40)
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print("\nClass frequencies in training set (top 10):")
    
    # Get class names if possible
    mapping_path = "/root/FINAL_CORRECTED_VERSION/label_mapping.csv"
    class_names = {}
    if os.path.exists(mapping_path):
        mdf = pd.read_csv(mapping_path)
        for _, row in mdf.iterrows():
            class_names[row['final_label_index']] = row['final_category']
    
    sorted_indices = np.argsort(-class_counts)
    for i in sorted_indices[:10]:
        name = class_names.get(i, f"Class {i}")
        print(f"  {name}: {int(class_counts[i])}")

    return {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'num_valid_classes': num_classes,
        'min_samples_threshold': min_samples
    }
