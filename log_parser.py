import re
import os
from collections import defaultdict
import multiprocessing as mp




def log_stats(params):
    status_dict=defaultdict(int)
    sources_dict=defaultdict(int)
    rgx = re.compile('([^ ]*) ([^ ]*) ([^ ]*) \[([^]]*)\] "([^"]*)" ([^ ]*) ([^ ]*)')
    filename, chunk_number, number_of_chunks = params
    with open(filename, 'r') as fp:
        for line in read_chunk(fp, number_of_chunks, chunk_number):
            status_dict[rgx.match(line).groups()[5]] += 1
            sources_dict[rgx.match(line).groups()[0]] += 1
    return status_dict,sources_dict



def read_chunk(fopen, number_of_blocks, block):
    '''
    A generator that splits a file into blocks and iterates
    over the lines of one of the blocks.

    '''

    fopen.seek(0, 2)
    file_size = fopen.tell()

    ini = file_size * block / number_of_blocks
    end = file_size * (1 + block) / number_of_blocks

    if ini <= 0:
        fopen.seek(0)
    else:
        fopen.seek(ini - 1)
        fopen.readline()

    while fopen.tell() < end:
        yield fopen.readline()


if __name__=='__main__':

    pool = mp.Pool()
    number_of_chunks=mp.cpu_count()
    filename='../../Users/sreek/Documents/wer_ai/case-studies/web.log'
    tasks= [(filename, i, number_of_chunks) for i in range(number_of_chunks)]
    status_results=pool.map(log_stats,tasks)
    print(status_results)









