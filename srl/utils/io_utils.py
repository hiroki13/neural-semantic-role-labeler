import sys


def say(s, stream=sys.stdout):
    stream.write(s + '\n')
    stream.flush()


def read_line_from_cmd():
    return raw_input('Input a sentence: ').rstrip()
