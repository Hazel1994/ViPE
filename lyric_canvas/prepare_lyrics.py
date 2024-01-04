import os
import json
from tqdm import tqdm
from utils import save_pkl, dotdict
from utils import preprocess_lyrics, preprocess_name_and_title
from string import digits
import argparse

remove_digits = str.maketrans('', '', digits)


def parse_args():
    parser = argparse.ArgumentParser(description="preprocess the lyrics and save them as a single dataset")

    parser.add_argument(
        "--path", type=str, required=True,
        help='where the scraper saved the data'
    )
    parser.add_argument(
        "--path_out", type=str, required=True,
        help='a directory to save the prepared lyrics as a single pickle file'
    )

    parser.add_argument(
        "--min_unique_word_per_line", type=int, default=2, help='# min number of unique words per line, otherwise omit'
    )
    parser.add_argument(
        "--max_line_length", type=int, default=20, help='max number of words per line, otherwise omit'
    )

    parser.add_argument(
        "--min_en_lyrics", type=int, default=20, help='minimum number of english song an artist should have'
    )
    parser.add_argument(
        "--min_tracks", type=int, default=50, help='min number of tacks an artist must have'
    )
    parser.add_argument(
        "--take_n_tracks", type=int, default=50, help='number of tracks we use from each artist'
    )
    parser.add_argument(
        "--min_line_per_track", type=int, default=15,
        help='lyrics should contain at least 15 lines with at least 4 unique words, used with '
             'the min_unique_word_per_line_in_track argument '
    )
    parser.add_argument(
        "--min_unique_word_per_line_in_track", type=int, default=4,
        help='lyrics should contain at least 15 lines with at least 4 unique words, used with min_line_per_track '
    )
    parser.add_argument(
        "--max_lines", type=int, default=50, help='max  number of lines per lyric: chatgpt gets confused with'
                                                  ' longer lyrics'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config = dotdict({})
    # min number of unique words per line, otherwise omit
    config.min_unique_word_per_line = args.min_unique_word_per_line
    # max number of words per line, otherwise omit
    config.max_line_length = args.max_line_length

    # some artist have very few english songs, so lets set a min limit for this
    config.min_en_lyrics = args.min_en_lyrics
    # min number of tacks an artist must have
    config.min_tracks = args.min_tracks
    #  number of tracks we use from each artist
    config.take_n_tracks = args.take_n_tracks
    # lyrics should contain at least 15 lines with at least 4 unique words
    config.min_line_per_track = args.min_line_per_track
    config.min_unique_word_per_line_in_track = args.min_unique_word_per_line_in_track
    # max  number of lines per lyric
    config.max_lines = args.max_lines
    # where the scraper saved the data
    path = args.path
    # where to save the dataset (just lyrics)
    path_out = args.path_out

    data = {}
    all_lines_count = 0
    all_songs_count = 0

    corrupted_files = ['Lit genius']
    skipped_tracks = 0

    for c, name in enumerate(tqdm(os.listdir(path))):

        with open('{}{}'.format(path, name)) as f:
            artist_collection = json.load(f)
        name = name[:-5]  # remove .json

        # some files contain garbage
        if name not in corrupted_files:
            name = preprocess_name_and_title(name)

            if len(artist_collection['songs']) >= config.min_tracks:
                data[name] = []

                for song in artist_collection['songs'][0:config.take_n_tracks]:

                    # first check the language and make sure it contains something
                    lyrics = song['lyrics'].split('\n')[1:]
                    # check for empty spaces
                    lyrics = [line for line in lyrics if line != ' ' and line != '']

                    if song['language'] == 'en' and len(lyrics) > 1:
                        song_data = {'title': 0, 'lyrics': 0}

                        # preprocess the lyrics
                        lyrics = preprocess_lyrics(lyrics, config)

                        if lyrics:
                            all_songs_count += 1
                            all_lines_count += len(lyrics)

                            title = song['full_title']
                            song_data['title'] = preprocess_name_and_title(title)
                            song_data['lyrics'] = lyrics

                            # add the lyric to the artist list
                            data[name].append(song_data)
                        else:
                            skipped_tracks += 1

                # remove artist if no/or not enough english lyrics were added
                if len(data[name]) < config.min_en_lyrics:
                    all_songs_count = all_songs_count - len(data[name])
                    del data[name]

    print(
        'processed probably less than {} number of lines from exactly {} songs :'.format(all_lines_count,
                                                                                         all_songs_count))
    print('skipped {} number of tracks(lyrics)'.format(skipped_tracks))

    save_pkl(data, path_out + 'dataset_' + str(config.max_lines))


if __name__ == "__main__":
    main()
