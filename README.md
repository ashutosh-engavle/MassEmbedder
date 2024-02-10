
# MassEmbedder

MassEmbedder is a Python tool designed to enrich large datasets with embeddings, leveraging OpenAI's API to generate vector representations for textual data. It's crafted to handle vast amounts of data efficiently, making it an invaluable asset for semantic analysis or enhancing datasets for machine learning models.

## Features

- **Scalability**: Breaks down large datasets into manageable chunks, processing them individually to ensure memory efficiency.
- **Parallel Processing**: Utilizes concurrent processing to speed up the embedding generation, making the tool suitable for large-scale datasets.
- **Robust Error Handling**: Implements retry logic for API calls and graceful error handling to ensure process reliability.
- **Efficient File Handling**: Skips already processed chunks to save time and resources, with an option to combine all chunks back into a single file after processing.

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/MassEmbedder.git
cd MassEmbedder
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare your dataset**: Ensure your data is in a CSV format and accessible to the script.
2. **Run `add_embeddings.py`**: This will start processing your data in chunks, adding embeddings to each row.
   ```bash
   python add_embeddings.py
   ```
3. **Combine chunks**: After processing, use `combine.py` to merge all chunks into a single file.
   ```bash
   python combine.py
   ```

## Configuration

- You can adjust the chunk size, parallelism level, and other parameters at the beginning of `add_embeddings.py`.
- Modify `combine.py` if you need to change the output directory or file naming conventions.

## Contributing

Contributions are welcome! If you have a feature request, bug report, or a suggestion, please open an issue or submit a pull request.

## License

MassEmbedder is open-sourced under the MIT license. See the LICENSE file for more details.
