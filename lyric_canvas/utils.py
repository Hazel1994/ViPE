import pickle
import string
from string import digits
import json
import os
import sys
import time
import numpy as np
import openai

remove_digits = str.maketrans('', '', digits)


def preprocess_name_and_title(text):
    """
    remove / from tha path and shorten the title if needed
    """
    text = text.replace('/', '-')
    if len(text.split(' ')) > 15:
        text = ' '.join(text.split(' ')[0:15])
    return text


def save_pkl(file, name):
    with open(name + '.pickle', 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(name):
    with open(name + '.pickle', 'rb') as handle:
        file = pickle.load(handle)
    return file


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def remove_non_ascii(a_str):
    ascii_chars = set(string.printable)

    return ''.join(
        filter(lambda x: x in ascii_chars, a_str)
    )


def preprocess_lyrics(lyrics, config):
    if 'You might also like' in lyrics:
        lyrics.remove('You might also like')

    # sanity check: lyrics should contain at least 10 lines with more than 3 unique words
    unique_pass_len = len(
        [i for i in lyrics if len(np.unique(i.split(' '))) >= config.min_unique_word_per_line_in_track])
    if unique_pass_len < config.min_line_per_track:
        return False

    new_lyrics = []

    for line in lyrics:
        # lets remove non ascii characters
        line = remove_non_ascii(line)

        # some Ads content that happen to leak in lyrics
        if 'You might also like' in line:
            line = line.replace('You might also like', ' ')

        line_len_unique = len(np.unique(line.split(' ')))
        line_len = len(line.split(' '))

        # check the number of word counts
        if line_len_unique >= config.min_unique_word_per_line and line_len <= config.max_line_length:
            new_lyrics.append(line)
    lyrics = new_lyrics

    # still contain 'min_line_per_track' number of lines?
    if len(lyrics) < config.min_line_per_track:
        return False

    # some lyrics contain '(Verse 1)', lets remove it
    lyrics[0] = lyrics[0].replace('(Verse 1)', '')

    # truncate the long lyrics because chatgpt gets confused
    if len(lyrics) > config.max_lines:
        lyrics = lyrics[0:config.max_lines]

    # remove 'Embed' that is attached to the last word in the last line
    elif 'Embed' in lyrics[-1]:
        # remove  23Embed or 1Embed from the last line
        last_line = lyrics[-1].split(' ')
        last_word = last_line[-1].replace('Embed', '').translate(remove_digits)
        last_line[-1] = last_word
        lyrics[-1] = ' '.join(last_line)
        # sometimes Embed is the only word
        if lyrics[-1] == ' ':
            lyrics.pop(-1)

    return lyrics


def generate_visual_elaboration(x, data_per_call):
    args = data_per_call['args']
    interval_status = data_per_call['interval_status']
    data = data_per_call['data']

    start_indx, end_indx = x[0], x[1]
    log_path = args.path_log_output
    openai.api_key = args.api_key
    path = args.path_data_output

    # system role
    f = open("system_role_v2.0", "r")
    system_role = f.read()

    failures = 0
    total_tokens = 0
    song_count = 1
    miss_aligned_count = 0
    response_time = []

    names = list(data.keys())

    # Create a file to write the output of the process to
    output_filename = f"{log_path}{start_indx}_{end_indx}_output.txt"

    start_all = time.time()
    print('\n')
    for c, name in enumerate(names[start_indx:end_indx]):
        songs = data[name]
        print('processing artist number ', c + 1, ' in interval ', x, ' out of ', len(names[start_indx:end_indx]))
        for song in songs:
            lyrics = song['lyrics']
            title = song['title']

            # create directory
            file_name = os.path.join(path, name, title)
            os.makedirs(os.path.dirname(file_name), exist_ok=True)

            # check if the output is already there
            if not os.path.isfile(file_name):

                # enumerate the lines
                song = [str(c + 1) + ". " + i for c, i in enumerate(lyrics)]
                song = '\n'.join(song)

                # get a very slow response from chatgpt
                start_i = time.time()

                messages = [
                    {"role": "user", "content": system_role},
                    # {"role": "user", "content": '\nPrioritize rule number 2 and don\'t use generic terms. Do you understand?'},
                    {"role": "assistant", "content": 'Yes, I understand. Let\'s get started!'},
                    {"role": "user", "content": song}
                ]

                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages
                    )

                except Exception as e:
                    print('exception: at  ', name, title, ' :', e)
                    # set the status to False
                    interval_status[x[0]] = False
                    output_file = open(output_filename, "a")
                    output_file.write('\n' + str(e) + '\n')
                    output_file.close()
                    return False

                end_i = time.time()
                time_elapsed_i = (end_i - start_i)
                response_time.append(time_elapsed_i)

                # save the response
                with open(file_name, 'w') as f:
                    json.dump(response, f)

                song_count += 1

                pred_len = len(response.choices[0].message.content.split('\n'))

                # hopefully we get back the same number of lines
                if pred_len != len(song.split('\n')):
                    miss_aligned_count += 1

                # check the stop reason
                for choice in response.choices:
                    if choice.finish_reason != 'stop':
                        print(choice.finish_reason)
                        failures += 1

                total_tokens += response.usage['total_tokens']

                original_stdout = sys.stdout  # Save a reference to the original standard output

                output_file = open(output_filename, "a")

                sys.stdout = output_file  # Change the standard output to the file we created.

                # print((c + 1), ' out of ', len(names[start_indx:end_indx]))

                if song_count % 5 == 0:
                    print('\naverage response time: {} seconds'.format(np.mean(response_time)))

                if song_count % 30 == 0:
                    print('start: {}, end: {}'.format(start_indx, end_indx))
                    print('total songs processed: {}'.format(song_count))
                    print('tokens usage : {} '.format(total_tokens))
                    print('cost : {} dollars '.format(total_tokens * 0.000002))
                    print('number of fail cases: {}'.format(failures))
                    print('number of miss aligned  cases: {}'.format(miss_aligned_count))
                    end_all = time.time()
                    time_elapsed = (end_all - start_all) / 60
                    print('time taken so far: {} mins'.format(time_elapsed))
                    print('________________________________________________')

                output_file.close()
                sys.stdout = original_stdout  # Reset the standard output to its original value

    # we are done with this interval
    interval_status[x[0]] = 'done'
    print('interval {}, is finished'.format(x))
    return True