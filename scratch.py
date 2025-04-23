from PIL import Image

from datasets import ALL_DATASETS

bad_datasets = []
good_datasets = []

for key, cls in ALL_DATASETS.items():
    print(f"Checking {key}")
    dataset = cls(
        val_dataset_dir="/home/oliver/datasets_drive/vpr_datasets",
        input_transform=None,
        which_set="test",
    )

    for i in range(len(dataset)):
        img, _ = dataset[i]
        if type(img) != Image.Image:
            bad_datasets.append(key)
            break
    else:
        good_datasets.append(key)

print(f"Total datasets: {len(ALL_DATASETS)}")
print(f"Bad datasets: {len(bad_datasets)}")
print(f"Good datasets: {len(good_datasets)}")
