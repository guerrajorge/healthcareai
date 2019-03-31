import pandas as pd
from pathlib import Path
import os
from uszipcode import SearchEngine
import numpy as np
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    options = parser.parse_args()

    dataset_dir = options.data_dir

    # national plan & provider enumeration system
    # wget https://data.cms.gov/api/views/85jw-maq9/rows.csv?accessType=DOWNLOAD
    nppes = pd.read_csv(Path(dataset_dir,
                             'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2016.csv'),
                        low_memory=False)

    columns_int = [x for x in nppes['Provider Type'].unique() if
                   'oncology' in str.lower(x) or 'dermatology' in str.lower(x)]
    prov_ix = nppes[nppes['Provider Type'] == 'Dermatology'].isin(columns_int).index

    dermatologies = nppes.loc[prov_ix][['National Provider Identifier', 'Provider Type',
                                        'Last Name/Organization Name of the Provider',
                                        'First Name of the Provider', 'Middle Initial of the Provider',
                                        'Credentials of the Provider', 'Gender of the Provider',
                                        'Street Address 1 of the Provider', 'City of the Provider',
                                        'Zip Code of the Provider', 'State Code of the Provider',
                                        'Country Code of the Provider']].drop_duplicates()

    columns = [x.replace(' ', '_') for x in dermatologies.columns]
    dermatologies.columns = columns

    search = SearchEngine()

    dermatologies['latitude'] = [0] * dermatologies.shape[0]
    dermatologies['longitude'] = [0] * dermatologies.shape[0]

    for zipc_ix, zipc in enumerate(dermatologies['Zip_Code_of_the_Provider']):
        if len(str(zipc)) > 5:
            zipc = int(str(zipc)[:5])
        elif len(str(zipc)) < 5:
            zipc = np.nan
        dermatologies.loc[zipc_ix, 'latitude'] = search.by_zipcode(zipc).lat
        dermatologies.loc[zipc_ix, 'longitude'] = search.by_zipcode(zipc).lng

    dermatologies.to_csv(Path(dataset_dir, 'Doctors4.csv'), index=False)
        

if __name__ == '__main__':
    main()
