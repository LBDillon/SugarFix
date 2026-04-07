import pandas as pd

# Read both CSV files
benchmark_df = pd.read_csv(
    '/Users/lauradillon/PycharmProjects/inverse_fold/Cleaned_research_flow/0_Main_data/Final_Paper_Folder/protein-design-bias/data/glyco_benchmark/manifests/benchmark_manifest_simple.csv'
)

proteinmpnn_df = pd.read_csv(
    '/Users/lauradillon/Downloads/proteinMPNN_results_all_chunks-2.csv'
)

print("Benchmark manifest shape before merge:", benchmark_df.shape)
print("ProteinMPNN results shape:", proteinmpnn_df.shape)
print("\nBenchmark columns:", benchmark_df.columns.tolist())
print("ProteinMPNN columns:", proteinmpnn_df.columns.tolist())

# Merge on uniprot_id (from benchmark) and Entry (from proteinmpnn)
merged_df = benchmark_df.merge(
    proteinmpnn_df,
    left_on='uniprot_id',
    right_on='Entry',
    how='left'
)

print("\nMerged shape:", merged_df.shape)
print("\nMerged columns:", merged_df.columns.tolist())

# Remove the redundant 'Entry' column since we have 'uniprot_id'
merged_df = merged_df.drop('Entry', axis=1)

# Write back to the benchmark manifest file
output_path = '/Users/lauradillon/PycharmProjects/inverse_fold/Cleaned_research_flow/0_Main_data/Final_Paper_Folder/protein-design-bias/data/glyco_benchmark/manifests/benchmark_manifest_simple.csv'
merged_df.to_csv(output_path, index=False)

print(f"\n✓ Merged file saved to: {output_path}")
print(f"Total rows in merged file: {len(merged_df)}")

# Show a sample
print("\nSample of merged data (first 5 rows with ProteinMPNN columns):")
print(merged_df[['uniprot_id', 'protein_name', 'sequence_score', 'entropy', 'mean_confidence']].head())
