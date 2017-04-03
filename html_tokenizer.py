# -*- coding: utf-8 -*-
"""
Custom tokenizer that parses HTML into tuples of tokens and its corresponding
entity in the following manner:
    - HTML tags are a token, eg. '<html>' >>> ('<html>', other_ent_type)
    - Anything inside of a tag is tokenized according to the parameters
    passed to the tagged_html_to_tuples function, with the exception of inline
    script and style tag content, which is tokenized in a similar way as
    a normal HTML tag with its contents being a part of the token as well,
    eg. '<style>body {background-color: linen;}</style>' >>>
            ('<style>body {background-color: linen;}</style>', other_ent_type)
"""

import re

# Regex used to remove whitespace
ws_remove = re.compile('\s+')

# Regex used to remove hyperlinks.
# This is useful because if the links are unique to each html file,
# they will count as another unique token, which increases
# the size of the dictionary used by the neural network for NER
# and therefore decreases performance.
href_remove = re.compile('href(?:\s*)=".+?"')

# Regex used to remove possible double tags and other unwanted tags,
# eg. '<LOCATION><strong>New York</strong></LOCATION>'
# would be parsed as <LOCATION>New York</LOCATION>
double_tag_remove = re.compile('<.+?>')


def tokenize_w_by_w(text: str, split_chars: tuple=None) -> list:
    split_chars = split_chars or (' ', '-', '(', ')', '[', ']')
    for c in split_chars:
        text = text.replace(c, ' {c} '.format(c=c))
    return text.split()


def tagged_html_to_tuples(doc: str, other_ent_type: str,
                          entities: list, char_by_char=False):
    # TODO documentation
    doc_tag_tuples = []

    doc = href_remove.sub('href=""', doc)
    doc = ws_remove.sub(' ', doc)

    char_iter = iter(doc)
    chars_to_write = ''

    for char in char_iter:
        if char == '<':
            # Start of a tag, possible entity
            # First write whatever is left in chars_to_write
            if len(chars_to_write):
                if char_by_char:
                    for c in chars_to_write:
                        doc_tag_tuples.append((c, other_ent_type))
                else:
                    tokens = tokenize_w_by_w(chars_to_write)
                    if len(tokens) > 0:
                        for token in tokens:
                            doc_tag_tuples.append((token, other_ent_type))
                            doc_tag_tuples.append((' ', other_ent_type))
                    # else:
                    #     It's a space or something like that
                        # for c in chars_to_write:
                        #     doc_tag_tuples.append((c, other_ent_type))
            chars_to_write = ''

            # Investigate tag content
            start = ''
            next_char = next(char_iter)
            while next_char != '>':
                start += next_char
                next_char = next(char_iter)

            tag_name = start.split()[0]

            if tag_name in entities:
                # Process found entity
                tag_inside = ''
                next_char = next(char_iter)
                while tag_name not in tag_inside:
                    tag_inside += next_char
                    next_char = next(char_iter)

                tag_inside = tag_inside.replace('</{}'.format(tag_name), '')

                # Remove possible unwanted tags
                tag_inside = double_tag_remove.sub('', tag_inside)

                if char_by_char:
                    for c in tag_inside:
                        doc_tag_tuples.append((c, tag_name))
                else:
                    tokens = tokenize_w_by_w(tag_inside)
                    if len(tokens) > 0:
                        for idx, token in enumerate(tokens):
                            doc_tag_tuples.append((token, tag_name))
                            if idx != len(tokens) - 1:
                                doc_tag_tuples.append((' ', tag_name))
                            else:
                                doc_tag_tuples.append((' ', other_ent_type))
                    else:
                        # It's a space or something like that
                        for c in tag_inside:
                            doc_tag_tuples.append((c, tag_name))

                if next_char != '>':
                    # Get rid of the rest of the end tag
                    end = ''
                    next_char = next(char_iter)
                    while next_char != '>':
                        end += next_char
                        next_char = next(char_iter)
            elif tag_name == 'style' or tag_name == 'script':
                # Handle possible inline style and script tag content
                tag_inside = ''
                next_char = next(char_iter)
                while next_char != '<':
                    tag_inside += next_char
                    next_char = next(char_iter)

                end = ''
                next_char = next(char_iter)
                while next_char != '>':
                    end += next_char
                    next_char = next(char_iter)

                doc_tag_tuples.append(('<{start}>{inside}<{end}>'.
                                       format(start=start,
                                              inside=tag_inside,
                                              end=end),
                                       other_ent_type))
            else:
                doc_tag_tuples.append(('<{}>'.format(start), other_ent_type))
        else:
            if char_by_char:
                if char == ' ':
                    chars_to_write += char
                    if len(chars_to_write):
                        for c in chars_to_write:
                            doc_tag_tuples.append((c, other_ent_type))

                    chars_to_write = ''
                else:
                    chars_to_write += char
            else:
                chars_to_write += char

    # Write whatever is left in chars_to_write
    if len(chars_to_write):
        if char_by_char:
            for c in chars_to_write:
                doc_tag_tuples.append((c, other_ent_type))
        else:
            tokens = tokenize_w_by_w(chars_to_write)
            if len(tokens) > 0:
                for token in tokens:
                    doc_tag_tuples.append((token, other_ent_type))
                    doc_tag_tuples.append((' ', other_ent_type))
            else:
                # It's a space or something like that
                for c in chars_to_write:
                    doc_tag_tuples.append((c, other_ent_type))

    return doc_tag_tuples


def html_to_tokens(doc: str, char_by_char=False):
    entities = []
    other_ent_type = ''

    return [t[0] for t in tagged_html_to_tuples(doc,
                                                other_ent_type=other_ent_type,
                                                entities=entities,
                                                char_by_char=char_by_char)]
