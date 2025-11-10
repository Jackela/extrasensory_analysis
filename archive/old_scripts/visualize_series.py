"""
Archived script: visualize first time series and create lightweight docs (English-only).

@module visualize_series
@deprecated Historical exploratory script; not part of the production pipeline.
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os

# Define file paths
PKL_FILE = 'database.pkl'
OUTPUT_IMAGE = 'timeseries_plot.png'
DOCS_FILE = 'PROJECT_DOCS.md'

def create_or_update_docs(dataset_name, data_shape, label):
    """
    Create or update the project documentation file with English-only content.

    @param {str} dataset_name - Name/key of the dataset sample visualized.
    @param {tuple} data_shape - Shape as (variables, time_points).
    @param {str} label - Label/metadata string for the dataset.
    @pre dataset_name is non-empty; data_shape is a 2-tuple of ints.
    @post PROJECT_DOCS.md reflects the latest dataset metadata and diagrams.
    """
    content = f"""
# PROJECT_DOCS.md

## 1. Project Overview
This repository explores a collection of 1053 multivariate time series stored in `database.pkl`.

## 2. Project Structure
```mermaid
graph TD
    A[database.pkl] --> B[Python Scripts];
    B --> C[Analysis & Visualization];
    C --> D[Models];
```

## 3. Core Components & Logic
### Data Structure
- `database.pkl`: a Python dict
- Top-level: `{{ 'dataset_name': <inner_dict> }}`
- Inner dict: `{{ 'data': <numpy.ndarray>, 'labels': <string> }}`

### Current Sample
- Dataset: `{dataset_name}`
- Shape (variables, time points): `{data_shape}`
- Label: `{label}`

## 4. Interaction & Data Flow
```mermaid
graph TD
    subgraph Data Loading
        A[Load database.pkl] --> B[Extract a sample dataset];
    end
    subgraph Visualization
        B --> C[Plot time series data];
        C --> D[Save plot as timeseries_plot.png];
    end
```
"""
    # Write if missing or changed (utf-8)
    if not os.path.exists(DOCS_FILE) or open(DOCS_FILE, 'r', encoding='utf-8').read() != content:
        with open(DOCS_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Project docs '{DOCS_FILE}' created/updated.")


def visualize_first_series():
    """
    Load the first time series from `database.pkl` and save a plot.

    @returns {None}
    @throws {FileNotFoundError} When `database.pkl` is missing.
    @post Saves `timeseries_plot.png` and updates `PROJECT_DOCS.md`.
    """
    try:
        with open(PKL_FILE, 'rb') as f:
            data = pickle.load(f)

        # Get the first dataset
        first_key = list(data.keys())[0]
        time_series_data = data[first_key]['data']
        label = data[first_key]['labels']

        # Update Documentation
        create_or_update_docs(first_key, time_series_data.shape, label)

        # Visualization
        print(f"Visualizing first dataset: '{first_key}'...")
        print(f"Shape (variables, time points): {time_series_data.shape}")

        num_variables, num_time_points = time_series_data.shape

        plt.figure(figsize=(15, 8))

        # Plot each variable
        for i in range(num_variables):
            plt.plot(time_series_data[i, :], label=f'Variable {i+1}')

        plt.title(f'Time Series Plot for: {first_key}')
        plt.xlabel('Time Points')
        plt.ylabel('Value')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plt.savefig(OUTPUT_IMAGE)
        print(f"Figure saved as '{OUTPUT_IMAGE}'.")

    except FileNotFoundError:
        print(f"Error: '{PKL_FILE}' not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    visualize_first_series()
