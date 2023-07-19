import recordlinkage as rl
from recordlinkage.datasets import load_febrl1
from recordlinkage.datasets import load_febrl4
from recordlinkage.index import Block


def test_annotation_link(tmp_path):
    path = tmp_path / "febrl_annotation_link.json"

    # get febrl4 file
    df_a, df_b, matches = load_febrl4(return_links=True)

    # get record pairs
    indexer = Block("given_name", "given_name")
    pairs = indexer.index(df_a, df_b)

    # create annotation file
    # write an annotation file for the Febrl4 dataset.
    rl.write_annotation_file(path, pairs[0:10], df_a, df_b)

    # read the result
    result = rl.read_annotation_file(path)

    assert result.links is None
    assert result.distinct is None


def test_annotation_dedup(tmp_path):
    path = tmp_path / "febrl_annotation_dedup.json"

    # get febrl4 file
    df_a, matches = load_febrl1(return_links=True)

    # get record pairs
    indexer = Block("given_name", "given_name")
    pairs = indexer.index(df_a)

    # create annotation file
    # write an annotation file for the Febrl4 dataset.
    rl.write_annotation_file(path, pairs[0:10], df_a)

    # read the result
    result = rl.read_annotation_file(path)

    assert result.links is None
    assert result.distinct is None
