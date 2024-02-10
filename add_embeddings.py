import gc
import os
import random
from openai import OpenAI
import pandas as pd
import time
from tqdm import tqdm
import concurrent.futures
from combine import combine

# from transformers import XLNetModel, XLNetTokenizer
# import torch
# # Load pre-trained model tokenizer
# tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
# # Load pre-trained model
# model = XLNetModel.from_pretrained('xlnet-base-cased')
# def get_embedding_with_XLNet(text):
#     # Encode text
#     input_ids = torch.tensor([tokenizer.encode(text)])

#     # Get hidden states
#     with torch.no_grad():
#         outputs = model(input_ids)
#         # The last hidden-state is the first element of the output tuple
#         last_hidden_states = outputs.last_hidden_state

#     # Take the mean of the last hidden state to get a single vector representation
#     embeddings = torch.mean(last_hidden_states, dim=1).squeeze()
    
#     return embeddings

client = OpenAI()
# Function to get embedding with retry logic
def get_embedding_with_openai(text, model="text-embedding-3-small", max_retries=1000000):
    retries = 0
    text = text.replace("\n", " ")
    delay = 1
    while retries < max_retries:
        try:
            # Attempt to get the embedding
            response = client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except Exception as e:
            retries += 1
            delay += random.random()
            time.sleep(delay)  # Sleep before retrying
    return None

def process_chunk(chunk, chunk_id, column_name_for_embedding='name'):
    # If chunk file exists, don't do any processing. 
    if os.path.isfile(f'Chunks/embeddings_chunk_{chunk_id}.csv'):
        return None, None, False

    # Initialize an empty list to store embeddings
    embeddings = []
    
    # Iterate over each row in the chunk with a tqdm progress bar
    for _, row in tqdm(chunk.iterrows(), total=chunk.shape[0], desc=f"Processing Chunk {chunk_id}", leave=False):
        # Get the embedding for the current row and append it to the embeddings list
        embedding = get_embedding_with_openai(row[column_name_for_embedding], model='text-embedding-3-large')
        embeddings.append(embedding)
    
    # Return the original chunk and the list of embeddings
    return chunk, embeddings, True

def write_embeddings(chunk, embeddings, chunk_id):
    # Ensure the chunk includes the embeddings
    chunk['Embeddings'] = embeddings

    chunk.to_csv(f'Chunks/embeddings_chunk_{chunk_id}.csv', mode='w', index=False, header=True)
    # chunk.to_hdf(f'embeddings_chunk_{chunk_id}.h5', key='data', mode='w')
    
    # Clean up to free memory
    del chunk
    gc.collect()

def generate_embeddings(input_file_name='amazon_dataset.csv', column_name_for_embedding = 'name', output_file_name='amazon_dataset_embeddings_large.csv', workers=10, chunk_size=36, save_dir='Chunks', combine_chunks=True):
    df = pd.read_csv(input_file_name)

    total_chunks = (len(df) + chunk_size - 1) // chunk_size  # Calculate total number of chunks
    os.makedirs(save_dir, exist_ok=True) # Make directory to save chunks

    # Use ThreadPoolExecutor to manage concurrent chunk processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # Create an overall progress bar
        overall_progress = tqdm(total=total_chunks, desc="Overall Progress", unit="chunk", position=0)

        # Submit each chunk processing as a separate future task
        future_to_chunk = {executor.submit(process_chunk, df.iloc[start:start + chunk_size], start // chunk_size, column_name_for_embedding): start for start in range(0, len(df), chunk_size)}

        # As each future completes, write its result
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_start = future_to_chunk[future]
            chunk_id = chunk_start // chunk_size
            try:
                chunk, embeddings, processed = future.result()
                if processed:
                    write_embeddings(chunk.copy(), embeddings, chunk_id)
                del future_to_chunk[future] # Important deletion as to not overwhelm the RAM with embedding results
            except Exception as exc:
                print(f'Chunk {chunk_id} starting at {chunk_start} generated an exception: {exc}')
            finally:
                overall_progress.update(1)  # Increase the progress bar by one as each task is completed

        overall_progress.close()  # Ensure the progress bar is closed at the end
        gc.collect()  # Collect garbage at the end of processing
    if combine_chunks:
        combine(path=save_dir, output_file=output_file_name, delete_chunk=True) # Combine all the chunks into a single output file

if __name__ == '__main__':
    generate_embeddings()