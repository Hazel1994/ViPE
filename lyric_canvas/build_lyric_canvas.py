import csv
import json
import os
import pickle
import re
import argparse
import pandas as pd

from tqdm import tqdm

pattern = r'^([1-9]|[1-4][0-9]|50)\.\s'  # Regular expression pattern

def parse_args():
    parser = argparse.ArgumentParser(description="Generate the LyricCanvas dataset")

    parser.add_argument(
        "--lyric_file", type=str,
        help='where the prepared lyric pickle-file is',
        default='/graphics/scratch2/staff/hassan/genuis_chatgpt/dataset_50.pickle'
    )
    parser.add_argument(
        "--ds_file", type=str,
        help='directory and file name of the dataset (.csv) to be saved',
        default='/graphics/scratch2/staff/hassan/genuis_chatgpt/test/lyric_canvas_complete.csv'
    )
    parser.add_argument(
        "--prompt_path", type=str,
        help='directory of chatgpt generated prompts or our preprocessed prompts file on hugging face',
        default='/graphics/scratch2/staff/hassan/genuis_chatgpt/test/lyric_canvas.csv'
    )
    parser.add_argument(
        "--vipe_prompts",  action='store_true',
        help='pass --vipe_prompts for using our preprocessed prompts on huggingface'
    )
    args = parser.parse_args()
    return args


def check_prompt_format(current_line):
    return bool(re.match(pattern, current_line))

# some prompts are reallt long like 60 or so, but only a few
def truncate(line, max_len):
    words = line.split(' ')
    if len(words) > max_len:
        words = words[0:max_len]
        return ' '.join(words)
    return line

# simplify gpt_ids
def gpt_id_simple(gpt_ids):
    unique_ids = list(set(gpt_ids))
    unique_ids = {v: k + 1 for k, v in enumerate(unique_ids)}
    new_ids = [unique_ids[id] for id in gpt_ids]

    return new_ids

def main():
    args = parse_args()
    ds_path=args.ds_file
    prompt_path=args.prompt_path

    with open(args.lyric_file, 'rb') as handle:
        file_lyrics = pickle.load(handle)

    # use the preprocessed prompts I made  available on hugginface
    if args.vipe_prompts:
        print('using the preprocessed prompts available on hugginface')
        # load the prompts from chatgpt
        prompts = pd.read_csv(prompt_path)

        lyrics_list = []
        lyrics_buffer=[]
        counter=0
        check_title=''
        artist_check=None
        progress_counter=0
        for  artist, song_name in tqdm(zip(prompts['artists'],prompts['titles'])):
            # artist = row['artists']
            # title = row['titles']
            # lyrics = row['lyrics']
            if progress_counter % 30000 == 0:
                print(progress_counter, 'out of ', len(prompts))

            keep_on=False
            progress_counter +=1


            while ((song_name in check_title) and  artist==artist_check):

                if len(lyrics_buffer) > counter:
                    lyrics_list.append(lyrics_buffer[counter])
                else:
                    lyrics_list.append('NAN')

                counter +=1
                keep_on=True
                break

            # keeo going according to the prompt file until the artist and song name changes
            # this is to make sure the length of the prompts and lyrics are the same
            if keep_on:
                continue

            #check if the artist exist in the lyrics file
            if artist in file_lyrics:
                songs=file_lyrics[artist]
                for song in songs:

                    if song_name in song['title']:
                        check_title = song['title']
                        artist_check=artist
                        counter=0
                        lyrics_buffer=song['lyrics']
                        lyrics_list.append(lyrics_buffer[counter])
                        counter +=1
                        break

        #fill in the lyrics
        prompts['lyrics']=lyrics_list
        prompts=prompts[prompts['lyrics'] !='NAN']
        prompts.to_csv(ds_path, index=False)
        print('done creating the dataset, total length: {}'.format(len(prompts)))


    else:# use the your your version of the prompts from an LLM

        lyric_prompt = {}
        my_id = []
        my_gpt_id = []
        my_artist = []
        my_song = []
        my_lyric = []
        my_prompt = []
        idx = 1
        size_threshold = 1000  # bytes
        max_prompt_len = 25  # more than 99% of the data have less than 25

        # c_bug=0
        for artist, songs in tqdm(file_lyrics.items()):

            # if c_bug >100:
            #     break

            # check if the artist exist
            if os.path.exists(os.path.join(prompt_path, artist)):

                for song in songs:

                    full_title = song['title']
                    lyric = song['lyrics']
                    # c_bug += 1

                    # check if prompts exist
                    if os.path.exists(os.path.join(prompt_path, artist, full_title)) == False or os.stat(
                            os.path.join(prompt_path, artist, full_title)).st_size < size_threshold:
                        continue
                    else:
                        with open(os.path.join(prompt_path, artist, full_title), 'r') as f:
                            gpt = json.load(f)  # json is faster than YAML

                    gpt_id = gpt['id'].split('chatcmpl-')[1]
                    prompts = gpt['choices'][0]['message']['content'].split('\n')

                    prompts = [truncate(x, max_prompt_len) for x in prompts if check_prompt_format(x)]

                    # prompts should contain something
                    if not prompts:
                        continue

                    # should also start with 1.
                    if '1. ' not in prompts[0]:
                        continue

                    # take the intersection of lyrics and prompts
                    for prom, line in zip(prompts, lyric):
                        my_id.append(idx)
                        my_gpt_id.append(gpt_id)
                        my_artist.append(artist)
                        my_song.append(full_title.split('by\xa0')[0])
                        my_lyric.append(line)
                        my_prompt.append(prom.split('.')[1])
                        idx = idx + 1

        lyric_prompt['ids'] = my_id
        lyric_prompt['gpt_ids'] = gpt_id_simple(my_gpt_id)
        lyric_prompt['artists'] = my_artist
        lyric_prompt['titles'] = my_song
        lyric_prompt['lyrics'] = my_lyric
        lyric_prompt['prompts'] = my_prompt

        keys = lyric_prompt.keys()
        with open(ds_path, "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(keys)
            writer.writerows(zip(*lyric_prompt.values()))

if __name__ == "__main__":
    main()
