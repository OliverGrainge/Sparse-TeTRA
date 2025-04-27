import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from dataloader.val import ALL_DATASETS
from prettytable import PrettyTable

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 
                'std': [0.5, 0.5, 0.5]}

TRAIN_CITIES = [
    'Bangkok',
    'BuenosAires',
    'LosAngeles',
    'MexicoCity',
    'OSL', # refers to Oslo
    'Rome',
    'Barcelona',
    'Chicago',
    'Madrid',
    'Miami',
    'Phoenix',
    'TRT', # refers to Toronto
    'Boston',
    'Lisbon',
    'Medellin',
    'Minneapolis',
    'PRG', # refers to Prague
    'WashingtonDC',
    'Brussels',
    'London',
    'Melbourne',
    'Osaka',
    'PRS', # refers to Paris
]

# https://github.com/amaralibey/gsv-cities

import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def pair(t):
    return (
        (t, t)
        if isinstance(t, int)
        else tuple(t[:2]) if isinstance(t, (list, tuple)) else (t, t)
    )


class GSVCitiesDataset(Dataset):
    def __init__(self,
                 cities=['London', 'Boston'],
                 img_per_place=4,
                 min_img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 base_path=None
                 ):
        super(GSVCitiesDataset, self).__init__()
        self.base_path = base_path
        self.cities = cities

        assert img_per_place <= min_img_per_place, \
            f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        
        # generate the dataframe contraining images metadata
        self.dataframe = self.__getdataframes()
        
        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)
        
    def __getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        # read the first city dataframe
        df = pd.read_csv(self.base_path+'Dataframes/'+f'{self.cities[0]}.csv')
        df = df.sample(frac=1)  # shuffle the city dataframe
        

        # append other cities one by one
        for i in range(1, len(self.cities)):
            tmp_df = pd.read_csv(
                self.base_path+'Dataframes/'+f'{self.cities[i]}.csv')

            # Now we add a prefix to place_id, so that we
            # don't confuse, say, place number 13 of NewYork
            # with place number 13 of London ==> (0000013 and 0500013)
            # We suppose that there is no city with more than
            # 99999 images and there won't be more than 99 cities
            # TODO: rename the dataset and hardcode these prefixes
            prefix = i
            tmp_df['place_id'] = tmp_df['place_id'] + (prefix * 10**5)
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe
            
            df = pd.concat([df, tmp_df], ignore_index=True)

        # keep only places depicted by at least min_img_per_place images
        res = df[df.groupby('place_id')['place_id'].transform(
            'size') >= self.min_img_per_place]
        return res.set_index('place_id')
    
    def __getitem__(self, index):
        place_id = self.places_ids[index]
        
        # get the place in form of a dataframe (each row corresponds to one image)
        place = self.dataframe.loc[place_id]
        
        # sample K images (rows) from this place
        # we can either sort and take the most recent k images
        # or randomly sample them
        if self.random_sample_from_each_place:
            place = place.sample(n=self.img_per_place)
        else:  # always get the same most recent images
            place = place.sort_values(
                by=['year', 'month', 'lat'], ascending=False)
            place = place[: self.img_per_place]
            
        imgs = []
        for i, row in place.iterrows():
            img_name = self.get_img_name(row)
            img_path = self.base_path + 'Images/' + \
                row['city_id'] + '/' + img_name
            img = self.image_loader(img_path)

            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)

        # NOTE: contrary to image classification where __getitem__ returns only one image 
        # in GSVCities, we return a place, which is a Tesor of K images (K=self.img_per_place)
        # this will return a Tensor of shape [K, channels, height, width]. This needs to be taken into account 
        # in the Dataloader (which will yield batches of shape [BS, K, channels, height, width])
        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)

    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def get_img_name(row):
        # given a row from the dataframe
        # return the corresponding image name

        city = row['city_id']
        
        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        pl_id = row.name % 10**5  #row.name is the index of the row, not to be confused with image name
        pl_id = str(pl_id).zfill(7)
        
        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        name = city+'_'+pl_id+'_'+year+'_'+month+'_' + \
            northdeg+'_'+lat+'_'+lon+'_'+panoid+'.jpg'
        return name
    



class PostTrainDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data_dir: str, 
                 val_data_dir: str,
                 batch_size=32,
                 img_per_place=4,
                 min_img_per_place=4,
                 shuffle_all=False,
                 image_size=(480, 640),
                 num_workers=4,
                 cities=TRAIN_CITIES,
                 mean_std=IMAGENET_MEAN_STD,
                 batch_sampler=None,
                 random_sample_from_each_place=True,
                 val_set_names=['pitts30k']
                 ):
        super().__init__()
        self.image_size = image_size
        self.train_data_dir = train_data_dir 
        self.val_data_dir = val_data_dir
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.shuffle_all = shuffle_all
    
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.cities = cities
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.random_sample_from_each_place = random_sample_from_each_place
        self.val_set_names = val_set_names
        self.save_hyperparameters() # save hyperparameter with Pytorch Lightening

        self.train_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])

        self.valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)])

        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': self.shuffle_all}

        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers//2,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False}
        
    def _transform(self):
        return T.Compose(
            [
                T.Resize(pair(self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def setup(self, stage):
        if stage == 'fit':
            self.reload()
            self.val_datasets = []
            for dataset_name in self.val_set_names:
                assert (
                    dataset_name.lower() in ALL_DATASETS.keys()
                ), f"Dataset {dataset_name} not found, must choose from {ALL_DATASETS.keys()}"

            self.val_datasets = [
                ALL_DATASETS[dataset_name](
                    val_dataset_dir=self.val_data_dir,
                    input_transform=self._transform(),
                    which_set="test",
                )
                for dataset_name in self.val_set_names
            ]

    def reload(self):
        self.train_dataset = GSVCitiesDataset(
            base_path=self.train_data_dir,
            cities=self.cities,
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform)

    def train_dataloader(self):
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self): 
        dataloaders = []
        for dataset in self.val_datasets:
            dataloaders.append(
                DataLoader(
                    dataset,
                    **self.valid_loader_config
                )
            )
        return dataloaders


    
   