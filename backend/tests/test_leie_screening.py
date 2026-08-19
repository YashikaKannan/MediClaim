import numpy as np
import pandas as pd

from app.pipeline import calculate_leie_scores, load_leie_npi_index


LEIE_PATH = "E:/CTS - MediClaim/datas/leie_clean_specialty_filled.csv"


def test_default_provider_ids_do_not_receive_fabricated_leie_hits():
    labels = pd.DataFrame({"Provider": ["PRV51001"], "PotentialFraud": [0]})

    scores = calculate_leie_scores(["PRV51001"], labels, LEIE_PATH)

    assert np.array_equal(scores, np.array([0.0]))


def test_exact_provider_npi_receives_leie_match_score():
    leie_npis = load_leie_npi_index(LEIE_PATH)
    sample_npi = next(iter(leie_npis))
    labels = pd.DataFrame({"Provider": ["PRV-TEST"], "NPI": [sample_npi]})

    scores = calculate_leie_scores(["PRV-TEST"], labels, LEIE_PATH)

    assert np.array_equal(scores, np.array([100.0]))
