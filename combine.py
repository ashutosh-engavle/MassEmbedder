import csv
import glob
import os

from tqdm import tqdm

def combine(path = 'Chunks', output_file = 'amazon_dataset_embeddings_large.csv', delete_chunk=False):
    all_files = glob.glob(os.path.join(path, "*.csv"))
    header_written = False

    with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = None

        for filename in tqdm(all_files, desc="Combining Chunks"):
            with open(filename, 'r', newline='', encoding='utf-8') as f_in:
                reader = csv.reader(f_in)
                header = next(reader)  # Read the header of the current file

                if not header_written:
                    writer = csv.writer(f_out)
                    writer.writerow(header)  # Write the header to the output file
                    header_written = True

                for row in reader:
                    writer.writerow(row)  # Write data rows

                f_out.flush()  # Flush after each file to ensure data is written
            
            if delete_chunk:
                os.remove(filename)

if __name__ == '__main__':
    combine()