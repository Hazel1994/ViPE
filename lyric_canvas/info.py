import argparse
import numpy as np
from utils import load_pkl


def parse_args():
    parser = argparse.ArgumentParser(description="general information about the lyrics")

    parser.add_argument(
        "--path_out", type=str, required=False, default='/graphics/scratch2/staff/hassan/genuis_chatgpt/',
        help='path to the prepared lyrics dataset'
    )
    parser.add_argument(
        "--max_lines", type=int,  required=False, default=50,
        help='max  number of lines per lyric used for preparing the lyrics'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    max_lines = args.max_lines
    path_out = args.path_out
    data = load_pkl(path_out + 'dataset_' + str(max_lines))

    print('number of artists: ', len(data))

    songs_count = 0
    lengths = []
    for name, songs in data.items():
        songs_count += len(songs)
        for song in songs:
            ll = len(song['lyrics'])
            lengths.append(ll)

    print('total songs: ', songs_count)
    print('total lines: ', sum(lengths))
    print('mean , std  lines per track: ', np.mean(lengths), ' ', np.std(lengths))
    print('min , max  lines per track: ', np.min(lengths), ' ', np.max(lengths))

    long_name_count = 0  # some files come with very long name, lets make sure we have shorted all of them
    for name in data.keys():
        if len(name.split(' ')) > 15:
            long_name_count += 1
        for song in data[name]:
            if len(song['title'].split(' ')) > 15:
                long_name_count += 1

                # if the name of a file or folder contains \ , it will create another directory while saving
                # lets make sure we dont have such names
            if '/' in song['title']:
                long_name_count += 1

        if '/' in name:
            long_name_count += 1

    print('number of bad name or title', long_name_count)


if __name__ == "__main__":
    main()

"""
expected output for our dataset
number of artists:  5549
total songs:  249948
total lines:  9909617
mean , std  lines per track:  39.646714516619454   10.775692430730166
min , max  lines per track:  14   50
number of bad name or title 0
"""