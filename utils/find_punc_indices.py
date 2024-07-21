from helpers import read_lines

tags_path = '/scratches/dialfs/alta/th624/exp-th624/Calibration_Uncertainty/data/output_vocabulary/labels.txt'

def find_punc_indices(tags_path):
    punc_indices = []
    tag_lines = read_lines(tags_path)
    
    for i, tag in enumerate(tag_lines):
        if tag.split('_')[0] == '$APPEND' and not any([tag.split('_')[1][i].isalpha() for i in range(len(tag.split('_')[1]))]):
            punc_indices.append(i)
    
        elif tag.split('_')[0] == '$TRANSFORM' and tag.split('_')[1] == 'CASE':
            punc_indices.append(i)
    
    return find_punc_indices


if __name__ == '__main__':
    find_punc_indices()
        