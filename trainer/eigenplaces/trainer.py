import os
from typing import List, Tuple, Union, Optional

import faiss
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .augmentations import (DeviceAgnosticColorJitter,
                           DeviceAgnosticRandomResizedCrop)
from .commons import InfiniteDataLoader
from .cosface_loss import MarginCosineProduct
from .datasets import EigenPlacesDataset, TestDataset

# Constants
RECALL_VALUES = [1, 5, 10, 20]
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]


def pair(x: Union[Tuple, any]) -> Tuple:
    """Convert single value to a pair if it's not already a tuple."""
    return x if isinstance(x, tuple) else (x, x)


def get_feature_dim(model: nn.Module, transform: T.Compose) -> int:
    """Determine the feature dimension of a model."""
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(img)
    return out.shape[1]


class EigenPlacesTrainer:
    """Trainer class for the EigenPlaces visual place recognition model.
    
    This trainer implements a multi-task learning approach with lateral and frontal views.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset_folder: Optional[str] = None,
        val_dataset_folder: Optional[str] = None,
        num_workers: int = 8,
        batch_size: int = 32, 
        model_lr: float = 0.00001,
        M: int = 15,
        N: int = 3,
        focal_dist: int = 10,
        min_images_per_class: int = 5,
        groups_num: int = 9,
        image_size: int = 224,
        iterations_per_epoch: int = 5000,
        s: float = 100,
        m: float = 0.4,
        fc_output_dim: Optional[int] = None,
        classifiers_lr: float = 0.01,
        lambda_lat: float = 1.0,
        lambda_front: float = 1.0,
        visualize_classes: bool = False,
        max_epochs: int = 1000,
        brightness: float = 0.7,
        contrast: float = 0.7,
        saturation: float = 0.7,
        hue: float = 0.5,
        random_resized_crop: float = 0.5,
        device: Optional[str] = None,
    ):
        """Initialize the EigenPlaces trainer.
        
        Args:
            model: Neural network model to train
            train_dataset_folder: Path to the training dataset
            val_dataset_folder: Path to the validation dataset
            num_workers: Number of worker processes for data loading
            batch_size: Batch size for training
            model_lr: Learning rate for the model optimizer
            M: Number of positive pairs per batch
            N: Number of negative pairs per batch
            focal_dist: Focal distance parameter
            min_images_per_class: Minimum number of images per class
            groups_num: Number of groups to use
            image_size: Size of the input images
            iterations_per_epoch: Number of iterations per epoch
            s: Scale factor for cosface loss
            m: Margin for cosface loss
            fc_output_dim: Output dimension of the fully connected layer
            classifiers_lr: Learning rate for classifier optimizers
            lambda_lat: Weight for lateral loss
            lambda_front: Weight for frontal loss
            visualize_classes: Whether to visualize classes
            max_epochs: Maximum number of epochs to train
            brightness: Brightness parameter for color jitter
            contrast: Contrast parameter for color jitter
            saturation: Saturation parameter for color jitter
            hue: Hue parameter for color jitter
            random_resized_crop: Scale parameter for random resized crop
            device: Device to use for training ('cuda' or 'cpu')
        """
        # Model parameters
        self.model = model
        self.model_lr = model_lr

        # Dataset parameters
        self.M = M
        self.N = N
        self.focal_dist = focal_dist
        self.min_images_per_class = min_images_per_class
        self.groups_num = groups_num
        self.train_dataset_folder = train_dataset_folder
        self.val_dataset_folder = val_dataset_folder
        self.visualize_classes = visualize_classes
        self.num_workers = num_workers
        self.batch_size = batch_size 

        # Training parameters
        self.image_size = image_size
        self.iterations_per_epoch = iterations_per_epoch
        self.max_epochs = max_epochs

        # Loss parameters
        self.s = s
        self.m = m
        self.fc_output_dim = fc_output_dim
        self.classifiers_lr = classifiers_lr

        # Multi-task loss weights
        self.lambda_lat = lambda_lat
        self.lambda_front = lambda_front

        # Augmentation parameters
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.random_resized_crop = random_resized_crop

        # Device setup
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # State tracking
        self._current_epoch = 0
        self._current_step = 0
        
        # Will be initialized during setup
        self.criterion = None
        self.model_optimizer = None
        self.train_transform = None
        self.groups = None
        self.classifiers = None
        self.classifiers_optimizers = None
        self.val_dataset = None
        self.val_recalls = None
        self.last_checkpoint_path = None

    def _setup_model(self) -> None:
        """Set up the model, loss function and optimizer."""
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.model_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.model_lr
        )
        self.train_transform = self._train_transform()
        self.val_transform = self._val_transform()

    def _setup_cosplace_groups(self) -> None:
        """Set up the EigenPlaces dataset groups."""
        self.groups = [
            EigenPlacesDataset(
                self.train_dataset_folder,
                M=self.M,
                N=self.N,
                focal_dist=self.focal_dist,
                current_group=n // 2,
                min_images_per_class=self.min_images_per_class,
                angle=[0, 90][n % 2],
                visualize_classes=self.visualize_classes,
            )
            for n in range(self.groups_num * 2)
        ]

    def _setup_group_classifiers(self) -> None:
        """Set up the classifiers and their optimizers for each group."""
        self.classifiers = nn.ModuleList([
            MarginCosineProduct(self.fc_output_dim, len(group), s=self.s, m=self.m)
            for group in self.groups
        ]).to(self.device)

        self.classifiers_optimizers = [
            torch.optim.Adam(classifier.parameters(), lr=self.classifiers_lr)
            for classifier in self.classifiers
        ]

    def _train_transform(self) -> T.Compose:
        """Create the training data transform pipeline."""
        return T.Compose([
            DeviceAgnosticColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
            ),
            DeviceAgnosticRandomResizedCrop(
                pair(self.image_size), 
                scale=[1 - self.random_resized_crop, 1]
            ),
            T.Normalize(
                mean=NORMALIZATION_MEAN, 
                std=NORMALIZATION_STD
            ),
        ])
    
    def _val_transform(self) -> T.Compose:
        """Create the validation data transform pipeline."""
        return T.Compose([
            DeviceAgnosticRandomResizedCrop(
                pair(self.image_size), 
                scale=[0.9999, 1]
            ),
            T.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ])

    def _setup_val_dataset(self) -> None:
        """Set up the validation dataset."""
        self.val_dataset = TestDataset(self.val_dataset_folder)

    def _checkpoint_setup(self) -> None:
        """Set up checkpoint tracking."""
        self.val_recalls = []
        self.last_checkpoint_path = None

    def train(self) -> None:
        """Train the model."""
        self._setup_model()
        self._setup_cosplace_groups()
        self._setup_group_classifiers()
        self._setup_val_dataset()
        self._checkpoint_setup()

        for epoch in range(self.max_epochs):
            self.train_epoch()
            recalls, recalls_str = self.val_epoch()
            print(f"Epoch {self._current_epoch}, {recalls_str}")
            self.checkpoint(recalls[0])

    def train_epoch(self) -> None:
        """Train for one epoch."""
        self.model.train()
        
        # Pick which pair of groups to use this epoch
        current_dataset_num = (self._current_epoch % self.groups_num) * 2

        # Create two infinite loaders (lateral & frontal)
        ds_lateral = InfiniteDataLoader(
            self.groups[current_dataset_num],
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        ds_frontal = InfiniteDataLoader(
            self.groups[current_dataset_num + 1],
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        ds_lateral_iter = iter(ds_lateral)
        ds_frontal_iter = iter(ds_frontal)

        for _ in tqdm(
            range(self.iterations_per_epoch),
            desc=f"Epoch {self._current_epoch}",
            total=self.iterations_per_epoch,
            leave=False,
        ):
            # Zero grads on the shared model optimizer
            self.model_optimizer.zero_grad()

            # ------ lateral step ------ #
            self._train_step(ds_lateral_iter, current_dataset_num)

            # ------ frontal step ------ #
            self._train_step(ds_frontal_iter, current_dataset_num + 1)

            # Finally update the shared model
            self.model_optimizer.step()

        # Advance epoch counter
        self._current_epoch += 1
    
    def _train_step(self, dataloader_iter, classifier_idx: int) -> None:
        """Perform a single training step with one classifier.
        
        Args:
            dataloader_iter: Iterator for the dataloader
            classifier_idx: Index of the classifier to use
        """
        # Zero grads on the classifier optimizer
        self.classifiers_optimizers[classifier_idx].zero_grad()
        
        # Pull a batch and move to device
        images, targets, _ = next(dataloader_iter)
        images = images.to(self.device)
        targets = targets.to(self.device)
        images = self.train_transform(images)
        
        # Forward + loss
        descriptors = self.model(images)
        output = self.classifiers[classifier_idx](descriptors, targets)
        loss = self.criterion(output, targets)
        
        # Backward
        loss.backward()
        
        # Step classifier
        self.classifiers_optimizers[classifier_idx].step()

    def val_epoch(self) -> Tuple[np.ndarray, str]:
        """Run validation for one epoch.
        
        Returns:
            Tuple containing:
                - numpy array of recall values
                - string representation of recalls for logging
        """
        self.model.eval()
        
        with torch.no_grad():
            # Process database images
            all_descriptors = self._extract_descriptors_from_subset(
                range(self.val_dataset.database_num)
            )
            
            # Process query images
            query_range = range(
                self.val_dataset.database_num,
                self.val_dataset.database_num + self.val_dataset.queries_num
            )
            self._extract_descriptors_to_array(
                all_descriptors, query_range
            )

        # Split descriptors by database and queries
        queries_descriptors = all_descriptors[self.val_dataset.database_num:]
        database_descriptors = all_descriptors[:self.val_dataset.database_num]

        # Use a kNN to find predictions
        recalls, recalls_str = self._compute_recalls(
            database_descriptors, queries_descriptors
        )
        
        return recalls, recalls_str
    
    def _extract_descriptors_from_subset(self, indices_range) -> np.ndarray:
        """Extract descriptors from a subset of the validation dataset.
        
        Args:
            indices_range: Range of indices to process
            
        Returns:
            Array containing all descriptors
        """
        # Create subset and dataloader
        subset_ds = Subset(self.val_dataset, list(indices_range))
        dataloader = DataLoader(
            dataset=subset_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )
        
        # Initialize output array
        all_descriptors = np.empty(
            (len(self.val_dataset), self.fc_output_dim), 
            dtype="float32"
        )
        
        # Process each batch
        for images, indices in tqdm(
            dataloader, 
            desc=f"Validation Epoch {self._current_epoch}", 
            total=len(dataloader), 
            leave=False
        ):
            images = images.to(self.device)
            images = self.val_transform(images)
            descriptors = self.model(images)
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.detach().cpu().numpy(), :] = descriptors
            
        return all_descriptors
    
    def _extract_descriptors_to_array(self, all_descriptors: np.ndarray, indices_range) -> None:
        """Extract descriptors to a pre-allocated array.
        
        Args:
            all_descriptors: Array to store descriptors in
            indices_range: Range of indices to process
        """
        # Create subset and dataloader
        subset_ds = Subset(self.val_dataset, list(indices_range))
        dataloader = DataLoader(
            dataset=subset_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )
        
        # Process each batch
        for images, indices in tqdm(
            dataloader, 
            desc=f"Validation Epoch {self._current_epoch}", 
            total=len(dataloader), 
            leave=False
        ):
            images = images.to(self.device)
            images = self.val_transform(images)
            descriptors = self.model(images)
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.detach().cpu().numpy(), :] = descriptors
    
    def _compute_recalls(
        self, 
        database_descriptors: np.ndarray, 
        queries_descriptors: np.ndarray
    ) -> Tuple[np.ndarray, str]:
        """Compute recall values for the validation dataset.
        
        Args:
            database_descriptors: Descriptors for the database images
            queries_descriptors: Descriptors for the query images
            
        Returns:
            Tuple containing:
                - numpy array of recall values
                - string representation of recalls for logging
        """
        # Use a kNN to find predictions
        faiss_index = faiss.IndexFlatL2(self.fc_output_dim)
        faiss_index.add(database_descriptors)
        _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))

        # For each query, check if the predictions are correct
        positives_per_query = self.val_dataset.get_positives()
        recalls = np.zeros(len(RECALL_VALUES))
        
        for query_index, preds in enumerate(predictions):
            for i, n in enumerate(RECALL_VALUES):
                if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break
                    
        # Convert to percentages
        recalls = recalls / self.val_dataset.queries_num * 100
        recalls_str = ", ".join(
            [f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)]
        )
        
        return recalls, recalls_str

    def checkpoint(self, recall: float) -> None:
        """Save a checkpoint of the model if it's the best so far. Delete the previous best."""
        print(f"Epoch {self._current_epoch}, {recall}")
        # Create checkpoint directory
        dirpath = f"checkpoints/EigenPlaces/{self.model.name}"
        os.makedirs(dirpath, exist_ok=True)

        is_best = not self.val_recalls or recall > max(self.val_recalls)

        if is_best:
            # Delete previous best checkpoint if it exists
            if self.last_checkpoint_path and os.path.exists(self.last_checkpoint_path):
                os.remove(self.last_checkpoint_path)
            
            # Save new best checkpoint
            self._save_checkpoint(dirpath, recall)

        # Track recall history
        self.val_recalls.append(recall)
        self.val_recalls = self.val_recalls[-10:]
    
    def _save_checkpoint(self, dirpath: str, recall: float) -> None:
        """Save a checkpoint of the model.
        
        Args:
            dirpath: Directory to save the checkpoint in
            recall: The recall value to use for the checkpoint filename
        """
        filename = os.path.join(
            dirpath, 
            f"{self.model.name}-Epoch-{self._current_epoch}-R@1-{recall:.2f}.pth"
        )
        torch.save(self.model.state_dict(), filename)
        self.last_checkpoint_path = filename
            
