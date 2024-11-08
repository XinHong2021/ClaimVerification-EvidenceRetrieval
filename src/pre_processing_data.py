"""
Data I/O functions
"""
import pathlib
from pathlib import Path
import pandas as pd
from pandas import DataFrame
import re
import json
from typing import List, Dict, Union, Tuple
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataclasses import dataclass
import random
from src.torch_utils import get_torch_device

# Colors for printing to the terminal
# Ref: https://stackoverflow.com/a/287944
class COLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


"""
Help with serialising sets to JSON
Ref: https://stackoverflow.com/a/8230505
"""
class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def load_from_json(names:list) -> List[dict]:
    DATA_PATH = pathlib.Path("data", "*")
    datasets = list()
    for name in names:
        with open(DATA_PATH.with_name(name).with_suffix(".json")) as json_file:
            dataset = json.load(json_file)
            datasets.append(dataset)
            print(f"Loaded {name}")
    return datasets

def load_as_dataframe(names:list, full_evidence:bool=False) -> List[DataFrame]:
    """
    Gets the json datasets as dataframes

    Args:
        names (list): List of filenames names minus the json suffix.
        full_evidence (bool): Whether to return the full evidence dataset
            as the final element.

    Returns:
        DefaultDict[str, pd.DataFrame]: Keys as names, values as the dataset
    """
    # Load datasets from json as dict
    datasets = load_from_json(names=names)

    # Process the names so that they are snake case
    var_names = [re.sub(pattern=r"-", repl="_", string=name) for name in names]

    # Name the datasets
    named_datasets = zip(var_names, datasets)

    # Find the evidence if available
    for var_name, dataset in named_datasets:
        if var_name == "evidence":
            df = DataFrame.from_dict(dataset, orient="index")
            df.columns = ["evidence_text"]
            locals()[var_name] = df

    # Parse each dataset into a DataFrame then join evidence if available
    datasets_df = list()
    for var_name, dataset in zip(var_names, datasets):
        if var_name == "evidence":
            continue
        df = DataFrame.from_dict(dataset, orient="index")
        if "evidence" in locals().keys():
            df = pd.merge(
                left=df.explode(column="evidences"),
                right=locals()["evidence"],
                how="left",
                left_on="evidences",
                right_index=True
            )
        df = df.reset_index(names=["claim"]) \
            .set_index(keys=["claim", "claim_text", "claim_label", "evidences"])
        datasets_df.append(df)

    # If required, attach the full evidence as the final element
    if full_evidence and "evidence" in locals():
        datasets_df.append(locals()["evidence"])

    return datasets_df

def slice_by_claim(
    dataset_df:DataFrame, start:int=None, end:int=1, labels:List[str]=None
) -> DataFrame:
    """
    Slice the dataset by a given start and end sequence index,
    optionally accept claim labels to filter by.

    Args:
        dataset_df (DataFrame): Dataset generated by `load_as_dataframe`.
        start (int, optional): Starting index. Defaults to None.
        end (int, optional): Ending index. Defaults to 1.
        labels (List[str], optional): Claim class labels. Defaults to None.

    Returns:
        DataFrame: A subset of `dataset_df`.
    """
    # Query by label if required
    if labels is not None:
        dataset_df = dataset_df.query(f"claim_label == {labels}")

    # Create the slice
    claim_slice = (
        dataset_df
        .loc[dataset_df.index.get_level_values("claim").unique()[start:end]]
    )
    return claim_slice

def get_claim_texts(dataset_df:DataFrame) -> List[str]:
    """
    Retrieve unique claim texts from a dataset.

    Args:
        dataset_df (DataFrame): Dataset generated by `load_as_dataframe`.

    Returns:
        List[str]: A list of claim texts.
    """
    return dataset_df.index.get_level_values("claim_text").unique().to_list()

def get_evidence_texts(dataset_df:DataFrame) -> List[str]:
    """
    Retrieve all the evidence texts from a dataset.

    Args:
        dataset_df (DataFrame): Dataset generated by `load_as_dataframe`.

    Returns:
        List[str]: A list of claim texts.
    """
    return dataset_df["evidence_text"].to_list()

def get_paired_texts(dataset_df:DataFrame) -> DataFrame:
    """
    Retrieve claim and evidence text from a dataset as ordered pair.

    Args:
        dataset_df (DataFrame): Dataset generated by `load_as_dataframe`.

    Returns:
        DataFrame: `index` is claim-evidence pairs, columns are
            `claim_text`, `evidence_text`.
    """
    claim_evidence_pairs = (
        dataset_df
        .reset_index()[["claim_text", "evidence_text"]]
    )
    return claim_evidence_pairs

def create_claim_output(
    claim_id:str,
    claim_text:str,
    claim_label:str="NOT_ENOUGH_INFO",
    evidences:List[str] = list()
) -> Dict[str, Dict[str, Union[str, List[str]]]]:
    """
    Generates a consistent claim object to write to JSON.

    Args:
        claim_id (str): The claim ID.
        claim_text (str): The claim text.
        claim_label (str, optional): Claim label. Defaults to "NOT_ENOUGH_INFO".
        evidences (List[str], optional): List of evidences. Defaults to list().

    Returns:
        Dict[str, Dict[str, Union[str, List[str]]]]: The claim dict.
    """
    claim_dict = {
        claim_id: {
            "claim_text": claim_text,
            "claim_label": claim_label,
            "evidences": evidences
        }
    }
    return claim_dict


@dataclass
class ClaimEvidencePair:
    claim_id:str
    evidence_id:str
    label:int = 0


class RetrievalDevEvalDataset(Dataset):
    """
    Dataset that can be used for running Eng evaluation on the
    `dev` set.
    """
    def __init__(
        self,
        dev_claims_path:Path=Path("./data/dev-claims.json"),
        evidence_path:Path=Path("./data/evidence.json"),
        pos_label:int=1,
        neg_label:int=0,
        n_neg_samples:int=10,
        seed:int=42,
        verbose:bool=True,
        device=None,
    ) -> None:
        super(RetrievalDevEvalDataset, self).__init__()
        self.verbose = verbose
        self.n_neg_samples = n_neg_samples
        self.seed = seed
        self.device = device if device is not None else get_torch_device()
        self.pos_label = pos_label
        self.neg_label = neg_label

        # Set the random seed
        random.seed(a=seed)

        # Load dev claims from json
        with open(dev_claims_path, mode="r") as f:
            self.claims = json.load(fp=f)

        # Load evidence library
        with open(evidence_path, mode="r") as f:
            self.evidence = json.load(fp=f)

        # Load data
        self.data = self.__generate_data()

        print(f"generated dataset n={len(self.data)}")
        return

    def __generate_data(self):
        # Generate a pool of negative samples
        neg_evidence_ids = [
            evidence_id
            for claim in self.claims.values()
            for evidence_id in claim["evidences"]
        ]
        random.shuffle(neg_evidence_ids)

        # Cumulator
        data = []

        for claim_id, claim in tqdm(
            iterable=self.claims.items(),
            desc="claims",
            disable=not self.verbose
        ):
            # Get positive samples
            pos_evidence_ids = claim["evidences"]

            for evidence_id in pos_evidence_ids:
                data.append(ClaimEvidencePair(
                    claim_id=claim_id,
                    evidence_id=evidence_id,
                    label=self.pos_label
                ))

            # Get negative samples
            n_picked = 0
            for neg_evidence_id in neg_evidence_ids:
                if n_picked >= self.n_neg_samples:
                    break
                if neg_evidence_id in pos_evidence_ids:
                    continue
                data.append(ClaimEvidencePair(
                    claim_id=claim_id,
                    evidence_id=neg_evidence_id,
                    label=self.neg_label
                ))
                n_picked += 1

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Union[str, torch.Tensor]]:
        text_pair = self.data[idx]

        claim_id = text_pair.claim_id
        claim_text = self.claims[claim_id]["claim_text"]

        evidence_id = text_pair.evidence_id
        evidence_text = self.evidence[evidence_id]

        label = torch.tensor(text_pair.label, device=self.device)

        return claim_text, evidence_text, label, claim_id, evidence_id

    @staticmethod
    def test():
        dataset = RetrievalDevEvalDataset()
        for i, data in enumerate(dataset):
            if i >= 5:
                break
            print(data)
        return


class RetrievalWithShortlistDataset(Dataset):
    """
    Dataset that can be used for retrieval classification
    when using a shortlist produced from the shortlisting stage.
    """
    def __init__(
        self,
        claims_paths:List[Path],
        claims_shortlist_paths:List[Path],
        evidence_path:Path=Path("./data/evidence.json"),
        inference:bool=False,
        claim_id:str=None,
        pos_label:int=1,
        neg_label:int=0,
        n_neg_samples:int=10,
        shuffle:bool=False,
        seed:int=42,
        verbose:bool=True,
        device=None,
    ) -> None:
        super(RetrievalWithShortlistDataset, self).__init__()
        self.verbose = verbose
        self.n_neg_samples = n_neg_samples
        self.shuffle = shuffle
        self.seed = seed
        self.device = device if device is not None else get_torch_device()
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.inference = inference

        # Set the random seed
        random.seed(a=seed)

        # Load train claims from json
        self.claims = dict()
        for train_claims_path in claims_paths:
            with open(train_claims_path, mode="r") as f:
                self.claims.update(json.load(fp=f))

        # Load train claims shortlist from json
        self.claims_shortlist = dict()
        for train_claims_shortlist_path in claims_shortlist_paths:
            with open(train_claims_shortlist_path, mode="r") as f:
                self.claims_shortlist.update(json.load(fp=f))

        # Load evidence library
        with open(evidence_path, mode="r") as f:
            self.evidence = json.load(fp=f)

        # Load data
        self.data = self.__generate_data(target_claim_id=claim_id)

        print(f"generated dataset n={len(self.data)}")
        return

    def __generate_data(self, target_claim_id:str=None):
        # Cumulator
        data = []

        for claim_id, claim in tqdm(
            iterable=self.claims.items(),
            desc="claims",
            disable=not self.verbose
        ):
            # This is to only return samples for one claim_id if desired
            if target_claim_id is not None and claim_id != target_claim_id:
                continue

            # Get positive samples
            if self.inference:
                pos_evidence_ids = []
            else:
                pos_evidence_ids = claim["evidences"]
                for evidence_id in pos_evidence_ids:
                    data.append(ClaimEvidencePair(
                        claim_id=claim_id,
                        evidence_id=evidence_id,
                        label=self.pos_label
                    ))

            # Generate a pool of negative samples from shortlist
            neg_evidence_ids = self.claims_shortlist.get(claim_id, [])
            if self.shuffle:
                random.shuffle(neg_evidence_ids)

            n_picked = 0
            for neg_evidence_id in neg_evidence_ids:
                if n_picked >= self.n_neg_samples:
                    break
                if neg_evidence_id in pos_evidence_ids:
                    continue
                data.append(ClaimEvidencePair(
                    claim_id=claim_id,
                    evidence_id=neg_evidence_id,
                    label=self.neg_label
                ))
                n_picked += 1

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Union[str, torch.Tensor]]:
        text_pair = self.data[idx]

        claim_id = text_pair.claim_id
        claim_text = self.claims[claim_id]["claim_text"]

        evidence_id = text_pair.evidence_id
        evidence_text = self.evidence[evidence_id]

        label = torch.tensor(text_pair.label, device=self.device)

        return claim_text, evidence_text, label, claim_id, evidence_id

    @staticmethod
    def test():
        dataset = RetrievalWithShortlistDataset(
            claims_paths=[
                Path("./data/train-claims.json"),
                Path("./data/dev-claims.json"),
            ],
            claims_shortlist_paths=[
                Path("./result/train_shortlist_evidences_max_500.json"),
                Path("./result/dev_shortlist_evidences_max_500.json"),
            ],
            n_neg_samples=999999999,
        )
        for i, data in enumerate(dataset):
            if i >= 5:
                break
            print(data)
        return


class LabelClassificationDataset(Dataset):
    """
    Dataset used for label classification.
    """

    LABEL_MAP = {
        "REFUTES": 0,
        "NOT_ENOUGH_INFO": 1,
        "SUPPORTS": 2,
        # "DISPUTED": # Excluded from training as it is not informative
    }

    def __init__(
        self,
        claims_paths:List[Path],
        evidence_path:Path=Path("./data/evidence.json"),
        claim_id:str=None,
        training:bool=False,
        verbose:bool=True,
        device=None,
    ) -> None:
        super(LabelClassificationDataset, self).__init__()
        self.verbose = verbose
        self.device = device if device is not None else get_torch_device()
        self.training = training

        # Load train claims from json
        self.claims = dict()
        for train_claims_path in claims_paths:
            with open(train_claims_path, mode="r") as f:
                self.claims.update(json.load(fp=f))

        # Load evidence library
        with open(evidence_path, mode="r") as f:
            self.evidence = json.load(fp=f)

        # Load data
        self.data = self.__generate_data(target_claim_id=claim_id)

        print(f"generated dataset n={len(self.data)}")

        pass

    def __generate_data(self, target_claim_id:str=None):

        data = []
        for claim_id, claim in tqdm(
            iterable=self.claims.items(),
            desc="claims",
            disable=not self.verbose
        ):
            # This is to only return samples for one claim_id if desired
            if target_claim_id is not None and claim_id != target_claim_id:
                continue

            # Get evidence ids associated with each claim
            evidence_ids = claim["evidences"]

            # Get the claim label
            # label = claim["claim_label"]
            label = claim.get("claim_label", "NOT_ENOUGH_INFO")

            # During training mode, exclude "DISPUTED"
            if self.training and label == "DISPUTED":
                continue

            # Encode the label
            # Default state is NEI
            label_encoding = self.LABEL_MAP.get(label, 1)

            # Assign the labels to each claim/evidence pairs
            for evidence_id in evidence_ids:
                data.append(ClaimEvidencePair(
                    claim_id=claim_id,
                    evidence_id=evidence_id,
                    label=label_encoding
                ))

            continue

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Union[str, torch.Tensor]]:
        # Fetch the required data rows
        data = self.data[idx]

        # Get the label
        label = torch.tensor(data.label, device=self.device)

        # Get text ids
        claim_id = data.claim_id
        evidence_id = data.evidence_id

        # Get text
        claim_text = self.claims[claim_id]["claim_text"]
        evidence_text = self.evidence[evidence_id]

        return claim_text, evidence_text, label, claim_id, evidence_id

    @staticmethod
    def test():
        dataset = LabelClassificationDataset(
            claims_paths=[
                Path("./data/train-claims.json"),
                # Path("./data/dev-claims.json"),
            ],
            training=True
        )
        for i, data in enumerate(dataset):
            if i >= 5:
                break
            print(data)
        return

class InferenceClaims(Dataset):
    """
    Dataset used for retrieval inference.
    """
    def __init__(self, claims_path:Path) -> None:
        super(InferenceClaims, self).__init__()
        with open(claims_path, mode="r") as f:
            self.claims = json.load(fp=f)
            self.claim_ids = list(self.claims.keys())
            print(f"loaded inference claims n={len(self.claim_ids)}")
        return

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx) -> Tuple[str]:
        claim_id = self.claim_ids[idx]
        claim_text = self.claims[claim_id]["claim_text"]
        return claim_id, claim_text

class RetrievedInferenceClaims(Dataset):
    """
    Dataset used for label inference.
    """
    def __init__(self, claims_path:Path) -> None:
        super(RetrievedInferenceClaims, self).__init__()
        with open(claims_path, mode="r") as f:
            self.claims = json.load(fp=f)
            self.claim_ids = list(self.claims.keys())
            print(f"loaded inference claims n={len(self.claim_ids)}")
        return

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx) -> Tuple[str]:
        claim_id = self.claim_ids[idx]
        claim_text = self.claims[claim_id]["claim_text"]
        evidences = self.claims[claim_id]["evidences"]
        return claim_id, claim_text, evidences


if __name__ == "__main__":
    # RetrievalDevEvalDataset.test()
    # RetrievalWithShortlistDataset.test()
    # LabelClassificationDataset.test()
    pass