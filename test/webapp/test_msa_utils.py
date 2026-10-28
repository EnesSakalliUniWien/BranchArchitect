import pytest
from msa_to_trees.pipeline import load_alignment

from webapp.services.msa.utils import msa_to_dict, process_msa_data


def test_process_msa_data_marks_unparseable_fasta_sequences_as_missing() -> None:
    result = process_msa_data(
        msa_content=">seq1\nACGT\n>seq2\n",
        num_trees=2,
        window_size=1,
        step_size=1,
    )

    assert set(result) == {"inferred_window_size", "inferred_step_size", "msa_dict"}
    assert result["msa_dict"] is None


def test_repeated_sequence_ids_are_rejected_not_overwritten() -> None:
    fasta = ">a\nACGT\n>a\nACGA\n>b\nAAAA\n"
    with pytest.raises(ValueError, match="repeats sequence ID 'a'"):
        msa_to_dict(fasta)
    with pytest.raises(RuntimeError, match="repeats sequence IDs: a"):
        load_alignment("", content=fasta)
