# Import the PDF partitioning function from the unstructured library
from unstructured.partition.pdf import partition_pdf

# Directory where the PDF is stored
output_path = "../content/"

# Full path to the PDF file
file_path = "/home/atharv/GenAI/study_vedya/langgraph/projects/chat_pdf/backend/data/attention_paper.pdf"

# Partition the PDF into structured elements (text, tables, images, etc.)
chunks = partition_pdf(
    # Path of the input PDF
    filename=file_path,

    # Extract tables while preserving their structure (rows and columns)
    infer_table_structure=True,

    # Use the high-resolution model.
    # Required for accurate table extraction and image detection.
    strategy="hi_res",

    # Extract image blocks from the PDF
    extract_image_block_types=["Image"],

    # If you want images saved to a folder instead of returned in memory,
    # uncomment the line below.
    # image_output_dir=output_path,

    # Store extracted images in the returned payload (Base64 encoded)
    # instead of saving them to disk.
    extract_image_block_to_payload=True,

    # Split the document into chunks based on headings/titles.
    chunking_strategy="by_title",

    # Maximum number of characters allowed in a chunk.
    max_characters=10000,

    # Merge very small chunks (<2000 characters) with neighboring chunks.
    combine_text_under_n_chars=2000,

    # Start a new chunk after approximately 6000 characters,
    # even if the maximum size has not been reached.
    new_after_n_chars=6000,
)

# Print the number of chunks extracted
print(f"Total chunks: {len(chunks)}")

# Print the first few chunks
for i, chunk in enumerate(chunks[:5]):
    print(f"\n----- Chunk {i+1} -----")
    print(chunk)