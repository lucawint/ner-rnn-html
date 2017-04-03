conll_fp = 'wikigold.conll.txt'
out_fp = conll_fp.replace('.txt', '_inline.txt')
sep_char = ' '

def tuples_to_inline(tuples: list):
    """
    Takes in tuples of tokens and tags, e.g.
    [('is', 'O'), ('the', 'O'), ('Empire', 'B-LOC'),
    ('State', 'I-LOC'), ('Building', 'I-LOC')]
    and transforms it into an inline tagged format:
    'is the <LOC>Empire State Building</LOC>'.
    
    :param tuples:  List of tuples of tokens and tags. 
    :return:        String with tags inline.
    """
    inline_str = ''
    other_entity = 'O'

    prev_ent_type = None
    entity_content = []

    tuples_iter = iter(tuples)

    for token, ent_type in tuples_iter:
        if prev_ent_type is None:
            # First word of the document
            if ent_type == other_entity:
                inline_str += '{}'.format(token)
            else:
                entity_content = [token]
            prev_ent_type = ent_type
        elif prev_ent_type != ent_type:
            # Different entity type than before

            # Handle possibly existing entity content
            if len(entity_content):
                inline_str += ' <{prev_ent_type}>{tokens}</{prev_ent_type}>'. \
                    format(prev_ent_type=prev_ent_type,
                           tokens=' '.join(entity_content))

            entity_content = []

            if ent_type == other_entity:
                inline_str += ' {}'.format(token)
            else:
                entity_content = [token]

            prev_ent_type = ent_type
        elif prev_ent_type == ent_type:
            # Same entity as before
            if prev_ent_type == other_entity:
                inline_str += ' {}'.format(token)
            else:
                entity_content.append(token)

    # In case anything is left
    if len(entity_content):
        if prev_ent_type != other_entity:
            inline_str += '<{prev_ent_type}>' \
                          '{tokens}' \
                          '</{prev_ent_type}>'. \
                format(prev_ent_type=prev_ent_type,
                       tokens=' '.join(entity_content))
        else:
            inline_str += ' {}'.format(' '.join(entity_content))

    return inline_str

with open(conll_fp) as conllfile,\
        open(out_fp, 'w') as outfile:
    # List of lists of tuples
    curr_doc_tuples = []
    for row in conllfile:
        if row == '-DOCSTART- O\n':
            # End of doc
            inline_str = tuples_to_inline(curr_doc_tuples)
            outfile.write('{}\n'.format(inline_str))

            curr_doc_tuples = []
        else:
            splitup = row.rstrip().split(sep_char)
            iob_tag = ''.join(splitup[1:])
            if len(iob_tag) == 1:
                tag = iob_tag
            else:
                tag = ''.join(iob_tag.split('-')[1:])
            token = splitup[0]
            if len(token):
                curr_doc_tuples.append((token, tag))
