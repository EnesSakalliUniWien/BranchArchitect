from webapp.services.msa.utils import process_msa_data


def test_process_msa_data_marks_unparseable_fasta_sequences_as_missing():
    result = process_msa_data(
        msa_content=">seq1\nACGT\n>seq2\n",
        num_trees=2,
        window_size=1,
        step_size=1,
    )

    assert set(result) == {"inferred_window_size", "inferred_step_size", "msa_dict"}
    assert result["msa_dict"] is None
