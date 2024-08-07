import pytesseract
from PIL import Image
import pdf2image
from huggingface_hub import InferenceClient
import json
import re
import pandas as pd
import fasttext.util


API_TOKEN = "hf_xxxxx" # replace with your API token

client = InferenceClient(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    token=API_TOKEN,
)

def OCR_extraction(pages):
    # Extract text from image
    text = []
    for page in pages:
        text.append(pytesseract.image_to_string(page))
    return text

def extract_line_items_llama(text):
    prompt_template = """Here's my OCR output from a pdf doc for the purpose of industrial quote generation. 
                        Generate a json file containing list of industrial product_name alphabetical only without special characters 
                        (eg. "Corner Bead paper faced 10"; "5 8 Firecode Core"; "Easy Clip S545") 
                        based on the parsed output (use common sense and imagining the parsed format from a table where related information may be parsed in separate lines), 
                        ignore the quantity (eg. 35,288.66) and unit (eg. 1,000 LF or 1,000 SF or 37.00 EA) in the parsed text:
                        
                        Only output something like this structured output without explanation and code:
                        [
                            {{ "product": "Product A"}},
                            {{ "product": "Product B"}},
                            {{ "product": "Product C"}}, 
                            ...
                        ]:
                        
                        {text}
                        """
    prompt = prompt_template.format(text=text)
    messages = [{"role": "system", "content": "You are a bot that responds only with the extracted product list."},
                {"role": "user", "content": prompt}]
    print("Waiting for Meta-Llama-3.1-8B-Instruct to generate output...")
    response = client.chat_completion(
        messages=messages,
        max_tokens=1650,
        stream=True,
        # temperature=0.0,
    )

    structured_line_items = ""
    for message in response:
        structured_line_items += message.choices[0].delta.content
    
    return structured_line_items.strip()


def extract_product_names(structured_line_items_str):
    try:
        # Use a regular expression to extract the product names
        product_names = re.findall(r'"product":\s*"(?:[^"\\]|\\.)*"', structured_line_items_str)
        # Remove the "product": and quotes from the matches
        product_names = [match[12:-1] for match in product_names]
        return product_names
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None


def compute_similarity_scores_fasttext(extracted_names, product_names):
    # Download and load the FastText model
    fasttext.util.download_model('en', if_exists='ignore')  # Download English model
    ft = fasttext.load_model('cc.en.300.bin')

    # Generate embeddings for the extracted names and product names
    extracted_embeddings = np.array([ft.get_sentence_vector(name) for name in extracted_names])
    product_embeddings = np.array([ft.get_sentence_vector(name) for name in product_names])
    
    # Compute cosine similarity scores
    similarity_scores = np.dot(extracted_embeddings, product_embeddings.T)
    
    return similarity_scores

def get_top_n_recommendations_fasttext(similarity_scores, product_names, n=10):
    recommendations = []
    for scores in similarity_scores:
        # Get the indices of the top-n scores
        top_indices = scores.argsort()[-n:][::-1]
        top_recommendations = [product_names[i] for i in top_indices]
        recommendations.append(top_recommendations)
    return recommendations


if __name__ == '__main__':
    # Convert PDF to images
    request_1 = pdf2image.convert_from_path('/Users/evanwyf/Downloads/AutoQuote/Data/Request-Response 1/Request 1.pdf')
    request_2 = pdf2image.convert_from_path('/Users/evanwyf/Downloads/AutoQuote/Data/Request-Response 2/Request 2.pdf')
    product_db = pd.read_csv('/Users/evanwyf/Downloads/AutoQuote/Data/ProductDB.csv')

    # Extract text from each image
    text = OCR_extraction(request_1)
    # Combine the extracted text into a single string
    combined_text = " ".join(text)

    # Extract the line items from the combined text
    structured_line_items_str = extract_line_items_llama(combined_text)
    if structured_line_items_str:
        print("Extracted Items:", structured_line_items_str)
        product_names = extract_product_names(structured_line_items_str)
        print("# of extracted products:", len(product_names))

        # Load the product database
        product_names_db = product_db.iloc[:, 2].tolist()  # Assuming the product name is in the third column
        print("# of DB products:", len(product_names_db))

        # Compute similarity scores using FastText
        similarity_scores = compute_similarity_scores_fasttext(product_names[0], product_names_db)

        # Get top-10 recommendations for each extracted product name
        top_10_recommendations = get_top_n_recommendations_fasttext(similarity_scores, product_names_db)
        for i, recommendations in enumerate(top_10_recommendations):
            print(f"Top-10 Recommendations for '{product_names[i]}': {recommendations}")
    else:
        print("Failed to extract line items.")
