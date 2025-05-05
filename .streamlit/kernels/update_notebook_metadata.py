import nbformat
import os

# Define the notebook path (updated to the correct location)
notebook_path = 'D:\\Documents\\GitHub\\CGABERT\\.streamlit\\kernels\\autocomplete_comparison.ipynb'

# Verify the notebook exists
if not os.path.exists(notebook_path):
    raise FileNotFoundError(f"Notebook not found at {notebook_path}")

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

# Define the kernel specification
kernel_spec = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.11.0"
    }
}

# Update the notebook metadata
notebook.metadata.update(kernel_spec)

# Save the updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(notebook, f)

# Verify the metadata was applied
with open(notebook_path, 'r', encoding='utf-8') as f:
    updated_notebook = nbformat.read(f, as_version=4)
    if 'kernelspec' in updated_notebook.metadata and updated_notebook.metadata['kernelspec']['name'] == 'python3':
        print(f"Successfully updated notebook metadata at {notebook_path}")
    else:
        raise ValueError(f"Failed to update notebook metadata at {notebook_path}")