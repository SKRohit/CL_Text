import csv
import json
import os

import datasets


_DESCRIPTION = """\
This is the contrastive learning IQT Comments dataset.
"""



# Name of the dataset usually match the script name with CamelCase instead of snake_case
class CLDataset(datasets.GeneratorBasedBuilder):
    """Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")
    def _info(self):
        # This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        features = datasets.Features(
            {
                "comments": datasets.Value("string"),
                "iqt_number": datasets.Value("int64"),
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="",
            # License for the dataset if available
            license="",
            # Citation for the dataset
            citation="",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # This method is tasked with defining the splits.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "csv_filepath": "./data/train_iqt_comments.csv",
                    "data_dir": "./data/",
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "csv_filepath": "./data/valid_iqt_comments.csv",
                    "data_dir": "./data/",
                    "split": "valid",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "csv_filepath": "./data/valid_iqt_comments.csv",
                    "data_dir": "./data/",
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(
        self, csv_filepath, data_dir, split  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """ Yields examples as (key, example) tuples. 
        
            csv_filepath: path to test/train/valid csv containing mp3 file name, song description and genre
            data_dir: test/train/valid directory containing all mp3 files
            split: name of the split, i.e. test, train or validation string 
        """
        with open(csv_filepath, 'r') as read_obj:
            csv_dict_reader = csv.DictReader(read_obj)
            for id_, row in enumerate(csv_dict_reader):
                yield id_, {
                    "comments": row["comments"],
                    "iqt_number": row["iqt_number"]
                }