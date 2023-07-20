"""Module for rendering and reading annotation files.

The annotation module provides functions and class for rendering
annotation files. Annotation files can be used in the browser-based
annotator application to label record pairs. The labelled dataset
can be used for training and validation of the record linkage process.
"""

import json

import numpy as np
import pandas as pd

SCHEMA_VERSION_LATEST = 1


def write_annotation_file(
    fp,
    pairs,
    df_a,
    df_b=None,
    dataset_a_name=None,
    dataset_b_name=None,
    *args,
    **kwargs,
):
    """Render and export annotation file.

    This function renders and annotation object and stores it in a
    json file. The function is a wrapper around the `AnnotationWrapper`
    class.

    Parameters
    ----------
    fp: str
        The path to the annotation file.
    pairs: pandas.MultiIndex
        The record pairs to annotate.
    df_a: pandas.DataFrame
        The data frame with full record information for the
        pairs.
    df_b: pandas.DataFrame
        In case of data linkage, this is the second data frame.
        Default None.
    dataset_a_name: str
        The name of the first data frame.
    dataset_b_name: str
        In case of data linkage, the name of the second data frame.
        Default None.

    """

    annotation_obj = AnnotationWrapper(
        pairs, df_a, df_b, dataset_a_name, dataset_b_name, *args, **kwargs
    )
    annotation_obj.to_file(fp)


def read_annotation_file(fp):
    """Read annotation file.

    This function can be used to read the annotation file
    and extract the results like the linked pairs and distinct
    pairs.

    Parameters
    ----------
    fp: str
        The path to the annotation file.

    Returns
    -------
    AnnotationResult
        An AnnotationResult object.

    Example
    -------
    Read the links from an annotation file::

        > annotation = read_annotation_file("result.json")
        > print(annotation.links)
    """
    return AnnotationResult.from_file(fp)


class AnnotationWrapper:
    """Annotation wrapper to render annotation file."""

    def __init__(
        self, pairs, df_a, df_b=None, dataset_a_name=None, dataset_b_name=None
    ):
        self.pairs = pairs
        self.df_a = df_a
        self.df_b = df_b
        self.dataset_a_name = dataset_a_name
        self.dataset_b_name = dataset_b_name

    def _get_value(self, df, loc_x, loc_y, *args, **kwargs):
        return self._cast_value(df.at[loc_x, loc_y], *args, **kwargs)

    @staticmethod
    def _cast_value(value, na_value=None):
        if pd.isnull(value):
            return na_value
        elif type(value).__module__ == np.__name__:
            return value.item()
        else:
            return value

    def _create_annotation(self):
        result = {"version": SCHEMA_VERSION_LATEST, "pairs": []}

        # transform multiindex into frame
        df_pairs = self.pairs.to_frame()

        if self.df_b is None:  # dedup
            df_b = self.df_a
            dataset_b_name = self.dataset_a_name
        else:  # link
            df_b = self.df_b
            dataset_b_name = self.dataset_b_name

        columns_a = list(self.df_a)

        for _index, pair in df_pairs.iterrows():
            result_record = {
                "fields": [],
                "identifiers": {
                    "a": {
                        "dataset": self._cast_value(self.dataset_a_name),
                        "record": self._cast_value(pair[0]),
                    },
                    "b": {
                        "dataset": self._cast_value(dataset_b_name),
                        "record": self._cast_value(pair[1]),
                    },
                },
            }

            # get the full data for this record
            for col in columns_a:
                result_record_field_a = {
                    "name": self._cast_value(col),
                    "value": self._get_value(self.df_a, pair[0], col),
                    "type": "String",
                }

                result_record_field_b = {
                    "name": self._cast_value(col),
                    "value": self._get_value(df_b, pair[1], col),
                    "type": "String",
                }

                result_record_field = {
                    "a": self._cast_value(result_record_field_a),
                    "b": self._cast_value(result_record_field_b),
                    "similarity": self._cast_value(None),
                }
                result_record["fields"].append(result_record_field)

            # append the result to a file
            result["pairs"].append(result_record)

        return result

    def to_file(self, fp):
        """Write annotation object to file.

        Parameters
        ----------
        fp: str
            The path to store the annotation file.
        """
        with open(str(fp), "w") as f:
            json.dump(self._create_annotation(), f, indent=2)


class AnnotationResult:
    """Result of (manual) annotation.

    Parameters
    ----------
    pairs: list
        Raw data of each record pair in the annotation file.
    version: str
        The version number corresponding to the file structure.

    """

    def __init__(self, pairs=None, version=SCHEMA_VERSION_LATEST):
        self.version = version
        self.pairs = pairs

    def _get_annotation_value(self, label, label_str=None):
        if self.pairs is None:
            return None

        result_pairs = []

        for item in self.pairs:
            label_value = item.get("label", None)

            if label_value == label:
                result_pairs.append(
                    (
                        item["identifiers"]["a"]["record"],
                        item["identifiers"]["b"]["record"],
                    )
                )

        if len(result_pairs) == 0:
            return None
        else:
            return pd.MultiIndex.from_tuples(result_pairs)

    @property
    def links(self):
        """Return the links.

        Returns
        -------
        pandas.MultiIndex
            The links stored in a pandas MultiIndex.
        """
        return self._get_annotation_value(1)

    @property
    def distinct(self):
        """Return the distinct pairs.

        Returns
        -------
        pandas.MultiIndex
            The distinct pairs stored in a pandas MultiIndex.
        """
        return self._get_annotation_value(0)

    @property
    def unknown(self):
        """Return the unknown or unlaballed pairs.

        Returns
        -------
        pandas.MultiIndex
            The unknown or unlaballed pairs stored in a pandas MultiIndex.
        """
        return self._get_annotation_value(None)

    def __repr__(self):
        return f"<Annotator pairs, version={self.version}>"

    @classmethod
    def from_dict(cls, d):
        """Create AnnotationResult from dict

        Parameters
        ----------
        d: dict
            The annotation file as a dict.

        Returns
        -------
        AnnotationResult
            An AnnotationResult object."""
        return cls(pairs=d["pairs"], version=d["version"])

    @classmethod
    def from_file(cls, fp):
        """Create AnnotationResult from file

        Parameters
        ----------
        fp: str
            The path to the annotation file.

        Returns
        -------
        AnnotationResult
            An AnnotationResult object."""
        with open(str(fp)) as f:
            content = json.load(f)

        return cls.from_dict(content)
