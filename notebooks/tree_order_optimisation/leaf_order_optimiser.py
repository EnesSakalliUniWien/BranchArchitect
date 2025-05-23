from pathlib import Path
from brancharchitect.leaforder.benchmark_utilities import profile_and_visualize
import matplotlib.pyplot as plt

# Set paths
data_dir = Path("notebooks/data")
output_dir = Path("output/benchmark_pdfs")
output_dir.mkdir(parents=True, exist_ok=True)

# Iterate through all files in the data directory
for file in data_dir.iterdir():
    if file.is_file():
        print(f"Processing {file.name}...")
        # Create a subfolder for this run
        run_output_dir = output_dir / file.stem
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Run your analysis (replace with your actual function)
        # This is an example; adapt as needed for your workflow
        df = profile_and_visualize(
            filepath=str(file), n_iterations=20, bidirectional=True
        )

        # Save the DataFrame as a PDF (example: using matplotlib table)
        fig, ax = plt.subplots(figsize=(12, min(0.5 * len(df), 20)))
        ax.axis("off")
        tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
        plt.tight_layout()
        pdf_path = run_output_dir / f"{file.stem}_profile.pdf"
        plt.savefig(pdf_path)
        plt.close(fig)
        print(f"Saved PDF to {pdf_path}")