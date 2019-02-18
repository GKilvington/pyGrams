import bz2
import os
import pickle

import pandas as pd

from scripts.utils.datesToPeriods import tfidf_with_dates_to_weekly_term_counts

if __name__ == '__main__':
    tfidf_base_filename = 'USPTO-random-1000.pkl.bz2'

    tfidf_filename = os.path.join('outputs', 'tfidf', tfidf_base_filename + '-tfidf.pkl.bz2')
    tfidf_data = pd.read_pickle(tfidf_filename)
    [tfidf_matrix, feature_names, document_week_dates, doc_ids] = tfidf_data

    term_counts_per_week, number_of_documents_per_week, week_iso_dates = tfidf_with_dates_to_weekly_term_counts(
        tfidf_matrix, document_week_dates)

    term_counts_data = [term_counts_per_week, feature_names, number_of_documents_per_week, week_iso_dates]
    term_counts_filename = os.path.join('outputs', 'termcounts', tfidf_base_filename + '-term_counts.pkl.bz2')
    os.makedirs(os.path.dirname(term_counts_filename), exist_ok=True)
    with bz2.BZ2File(term_counts_filename, 'wb') as pickle_file:
        pickle.dump(term_counts_data, pickle_file, protocol=4)
