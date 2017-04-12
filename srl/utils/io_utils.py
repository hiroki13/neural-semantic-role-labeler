import sys


def say(s, stream=sys.stdout):
    stream.write(s + '\n')
    stream.flush()

