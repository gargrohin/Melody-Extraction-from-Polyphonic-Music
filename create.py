import os
from generate import generate

'''
Code to preprocess the dataset.
Uses the generate function from generate.py
Stores the Power Spectrum of wav files from the dataset as a npz file
'''

def preprocess():
    ground_path = '/mnt/Stuff/Acads/UGP/mycode/ground'
    ground_list = [x.split('.')[0].strip() for x in os.listdir(ground_path)]
    medleydb_path = '/mnt/data/datasets/medleyDB/V1'
    dest = '/mnt/data/datasets/stuffs'
    dest_voc = '/mnt/data/datasets/stuffs_voc'
    dest_bg = '/mnt/data/datasets/stuffs_bg'
    mir_voc = '/mnt/data/datasets/MIR_voc'
    mir_bg = '/mnt/data/datasets/MIR_bg'
    violin = '/mnt/data/datasets/Bach10_v1.1'
    dest_violin = '/mnt/data/datasets/stuffs_bach'
    dest_bachbg = '/mnt/data/datasets/stuffs_bachbg'

    # fh = open('outs.txt','r')
    # ad = [x.strip()[4:-4] for x in fh]
    #
    # for song in os.listdir(medleydb_path):
    #     if song in ground_list:
    #         if song in ad:
    #             #print('fine\n')
    #             continue
    #         song_folder = os.path.join(medleydb_path,song)
    #         for wav in os.listdir(song_folder):
    #             if '.wav' in wav:
    #                 generate(os.path.join(song_folder,wav))

    for song in os.listdir(violin):
        # if song.split('_')[0] != 'amy':
        #     continue
        generate(os.path.join(os.path.join(violin,song) , song + '-saxphone' + '.wav'), dest_bachbg)
        generate(os.path.join(os.path.join(violin,song) , song + '-clarinet' + '.wav'), dest_bachbg)
        generate(os.path.join(os.path.join(violin,song) , song + '-bassoon' + '.wav'), dest_bachbg)
    # for song in os.listdir(mir_bg):
    #     # if song.split('_')[0] != 'amy':
    #     #     continue
    #     generate(os.path.join(mir_bg,song), dest_bg)

if __name__ == '__main__':
    preprocess()
