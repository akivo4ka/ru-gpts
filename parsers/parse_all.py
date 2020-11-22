#!/usr/bin/env python
# coding: utf-8

from parsers import b17_parser, psy_files_parser, psy_practice_parser


def read_file(filename):
    with open(filename, 'rb') as f:
        text = f.read()
    print(len(text), filename)
    return text


def parse_and_save():
    # for parser in [b17_parser, psy_files_parser, psy_practice_parser]:
    #     parser.parse_and_save()

    train_text = b''
    for filename in ['b17.txt', 'psy_files.txt', 'psy_practice.txt']:
        train_text += read_file(filename)

    with open('all.txt', 'wb') as f:
        f.write(train_text)


if __name__ == "__main__":
    parse_and_save()
