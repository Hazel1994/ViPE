import argparse
import time
from multiprocessing import Manager
from pathos.multiprocessing import ProcessingPool as Pool
from utils import generate_visual_elaboration, load_pkl


def parse_args():
    parser = argparse.ArgumentParser(description="generate prompts for preprocessed lyrics!")

    parser.add_argument(
        "--n_chunks", type=int, default=100, help='number of parallel calls to ChatGPT'
    )

    parser.add_argument(
        "--path_data_output", type=str, default='/graphics/scratch2/staff/Hassan/chatgpt_data_v2.0/',
        help='a directory to save the results'
    )

    parser.add_argument(
        "--path_data_input", type=str, default='/graphics/scratch2/staff/Hassan/genius_crawl/dataset_50',
        help='the pickle file from the prepare_lyrics.py'
    )

    parser.add_argument(
        "--path_log_output", type=str, default='/graphics/scratch2/staff/Hassan/logs/chatgpt/',
        help='each thread will write its progress in a separate file'
    )

    parser.add_argument(
        "--api_key", type=str, required=True, default="sk-???"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    data = load_pkl(args.path_data_input)
    max_len = len(data.keys())

    start_indx = 0
    data = {k: data[k] for k in list(data.keys())[0:max_len]}

    interval_len = int((max_len - start_indx) / args.n_chunks)

    intervals_1 = list(range(start_indx, max_len, interval_len))
    intervals_2 = list(range(start_indx + interval_len, max_len, interval_len))
    if intervals_2[-1] + 1 != max_len:
        intervals_2.append(max_len)

    arguments = list(zip(intervals_1, intervals_2))
    print(arguments[0])
    print(arguments[-1])
    manager = Manager()
    interval_status = manager.dict({a: True for a, _ in arguments})

    data_per_call = {'args': args, 'interval_status': interval_status, 'data': data}

    def my_func(current_interval):
        return generate_visual_elaboration(current_interval, data_per_call)

    pool = Pool(len(arguments))
    pool.amap(my_func, arguments)

    go_on = True

    while go_on:
        time.sleep(120) # wait for 2 mins and check the status
        finished = 0
        for c, (key, status) in enumerate(interval_status.items()):
            if status is False:
                print('failed for {} interval trying again..'.format(arguments[c]))

                # set the status back to True
                interval_status[key] = True
                # resume the process
                pool.amap(my_func, [arguments[c]])
            # pools[c].amap(my_func, [arguments[c]])
            elif status == 'done':
                finished += 1

        if finished == len(arguments):
            go_on = False
            print('successfully finished all the chunks')

        if finished > 0:
            print('{} intervals are done out of {}'.format(finished, len(arguments)))

    pool.close()
    pool.join()
    pool.terminate()

if __name__ == "__main__":
    main()
