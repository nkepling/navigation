import argparse
import sys
import os

def parse_args():
    # Handle file-based arguments
    new_args = []
    for arg in sys.argv[1:]:
        if arg.startswith('@'):
            with open(arg[1:], 'r') as f:
                # Add each line in the file to the arguments list, splitting by whitespace
                new_args.extend(f.read().split())
        else:
            new_args.append(arg)
    return new_args




if __name__ == "__main__":
    pass

