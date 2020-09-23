from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
import argparse

def bert_embed(output_file, input_file):
    # Load pre-trained Sentence Transformer Model (based on DistilBERT). It will be downloaded automatically
    model = SentenceTransformer('distiluse-base-multilingual-cased')

    # Embed a list of sentences (from file)
    with open(input_file, "r", encoding='utf-8-sig') as fin:
        lines = fin.readlines()
        
    embed_result = open(output_file, "wb")
    
    model.encode(lines, show_progress_bar: bool = True).tofile(embed_result)

    embed_result.close()


def _main():
    parser = argparse.ArgumentParser('Create file with sentence embeddings for further prccessing.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--inputs', type=str, 
                        help='input text file')

    parser.add_argument('-o', '--output', type=str,
                        help='output file with sentence embeddings')

    #### Just some code to print debug information to stdout
    np.set_printoptions(threshold=100)

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout


    args = parser.parse_args()
    bert_embed(output_file=args.output,
       input_file=args.inputs)


if __name__ == '__main__':
    _main()