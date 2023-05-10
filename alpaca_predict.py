'''
Generate output predictions file with ids using the alpaca model
'''

import argparse
import sys
import os

from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        required=True)
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        required=True)
    args = parser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/alpaca_predict.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    model_name = f"chavinlo/alpaca-13b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # test

    text = 'Grammatically correct the following sentence. He is run to park.'

    inputs = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    out = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    import pdb; pdb.set_trace()
