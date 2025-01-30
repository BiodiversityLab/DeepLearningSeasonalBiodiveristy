import pandas as pd

data_dir = 'data/IBA-SE-2019-Malaise-trap-data/'

tax_level='Phylum'
tax_name='Arthropoda'

cluster_counts = pd.read_csv(data_dir + 'CO1_cleaned_nochimera_cluster_counts_SE_2019_PRESENCE_ABSENCE.tsv', sep='\t', header=0, index_col='cluster')
cluster_taxonomy = pd.read_csv(data_dir + 'CO1_cleaned_nochimera_cluster_taxonomy_SE_2019.tsv', sep='\t', header=0, index_col='Unnamed: 0')
samples_metadata = pd.read_csv(data_dir + '16S_CO1_malaise_samples_metadata_SE_2019_cleaned.tsv', sep='\t', header=0)
bio_spikeins = pd.read_csv(data_dir + 'malaise_bio_spikeins_SE_2019.tsv', sep=' ', quotechar='"')

# Get a subset of taxa according to taxonomic level and name
taxa = cluster_taxonomy[cluster_taxonomy[tax_level] == tax_name]

taxa['Order'].value_counts()
taxa_targeted = cluster_counts.loc[taxa.index]

# Remove taxa that was used as reference (i.e. artificially added)
remove_taxa = bio_spikeins['cluster'].drop_duplicates()
taxa_targeted = taxa_targeted[~taxa_targeted.index.isin(remove_taxa)].T

taxa_data = taxa_targeted.join(samples_metadata.set_index('sampleID_LAB'))

# Remove all non-lysate samples
taxa_data = taxa_data[taxa_data['lab_sample_type'] == 'lysate']

# Remove all samples without coordinates
taxa_data = taxa_data.dropna(subset=['latitudeWGS84', 'longitudeWGS84'])

# Covert to datetime
taxa_data['date_placed'] = pd.to_datetime(taxa_data['placing_date'])
taxa_data['date_collected'] = pd.to_datetime(taxa_data['collecting_date'])

# ADD UNIQUE ID SUFFIX BASED ON TRAP AND SORTED ON PLACING DATE
# Sort by date
taxa_data = taxa_data.sort_values(by='date_placed')

# Add suffix based on group sequence
taxa_data['unique_id'] = taxa_data.groupby('trapID').cumcount() + 1
taxa_data['unique_id'] = taxa_data['trapID'] + '_' + taxa_data['unique_id'].astype(str)

# Calculate the sum of taxa per sample and join
sample_taxa_sums = pd.DataFrame(taxa_targeted.T.iloc[:, 1:].sum()).rename(columns={0: 'n_taxa'})
taxa_data = taxa_data.join(sample_taxa_sums)

# Covert to datetime and calculate middle day and days the trap was set up
taxa_data.loc[:, 'middle_date'] = taxa_data['date_placed'] + (taxa_data['date_collected'] - taxa_data['date_placed']) / 2
taxa_data['days_up'] = (taxa_data['date_collected'] - taxa_data['date_placed']).dt.days

# Subset to date
start = pd.to_datetime('2019-04-01')
end = pd.to_datetime('2019-09-30')
taxa_data_subset = taxa_data[(taxa_data['date_placed'] >= start) & (taxa_data['date_collected'] <= end)]

taxa_data_subset = taxa_data_subset[['trapID', 'unique_id', 'n_taxa', 'date_placed', 'date_collected', 'middle_date', 'days_up']]

# # Drop trap TSKEKT because it does have bad values for the bioclim variables
taxa_data_subset = taxa_data_subset[taxa_data_subset['trapID'] != "TSKEKT"]
# # bad values for era5
taxa_data_subset = taxa_data_subset[taxa_data_subset['trapID'] != "TTAPUR"]
# # bad values for ph
taxa_data_subset = taxa_data_subset[taxa_data_subset['trapID'] != "T6YFBI"]
# # bad values for ph and hii
taxa_data_subset = taxa_data_subset[taxa_data_subset['trapID'] != "T1YMRX"]

#taxa_data_subset.to_csv('data/train_data.csv', index_label=False)

example_data = taxa_data_subset[taxa_data_subset['trapID'].isin(taxa_data_subset['trapID'].unique()[:2])]
example_data.to_csv('data/EXAMPLE_DATA.csv', index_label=False)
