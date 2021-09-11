import os
import argparse
import matplotlib.pyplot as plt

import timit_utils as tu
import timit_utils.audio_utils as au
import timit_utils.drawing_utils as du

# Visualize one sample and save plots
def Visualize(args):
    corpus = tu.Corpus(args.data_path)
    train = corpus.train 
    corpus.train, corpus.test

    scale_wav = args.scale_wav
    scale_words = args.scale_words
    scale_phones = args.scale_phones
    phn = args.phn
    idx = args.idx
    
    if args.save:
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)
    
    sentence = train.sentences_by_phone_df(phn).sentence[idx]
    print(f'Sentence {sentence.name}, sample rate: {sentence.sample_rate}')
    
    du.DrawVerticalPanels([du.AudioPanel(sentence.raw_audio, show_x_axis=True)], scale=scale_wav)
    if args.save: plt.savefig(args.out_path + "/waveform_phn_"+phn+"_idx_" + str(idx) +".png")
    
    du.DrawVerticalPanels([du.WordsPanel(sentence.words_df, sentence.raw_audio.shape[0], 
                                                        show_x_axis=True)], scale=scale_words)
    if args.save: plt.savefig(args.out_path + "/words_phn_"+phn+ "_idx_" + str(idx) +".png")

    du.DrawVerticalPanels([du.PhonesPanel(sentence.phones_df, sentence.raw_audio.shape[0])],
                                                                  scale = scale_phones)
    if args.save: plt.savefig(args.out_path + "/phones_phn_"+phn+ "_idx_" + str(idx) +".png")
    
    if args.show: plt.show()


# Visualize one sample, its characteristics and features and save plots
def Visualize_features(args):
    corpus = tu.Corpus(args.data_path)
    train = corpus.train 
    corpus.train, corpus.test

    scale_wav = args.scale_wav
    scale_words = args.scale_words
    scale_phones = args.scale_phones
    phn = args.phn
    idx = args.idx
    
    # Get a dataframe containing word counts spoken by all people.
    words_counts_df = train.words_to_usages_df

    # Get a dataframe containing phoneme counts spoken by all people.
    phones_counts_df = train.phones_to_usages_df.head(3)

    # Region
    #print(train.regions)

    # Region by index
    r_idx = train.region_by_index(idx)
    print("Region by index: ", r_idx.name)
    
    # Region by index and Person from region    
    r_idx = train.region_by_index(idx).person_by_index(idx)
    print("Person from region by index: ", r_idx.name)
    
    # Person by index
    p_idx = train.person_by_index(idx) 
    print("Person by index: ", p_idx.name)
    #print(p_idx.sentences)

    # Sentence by index
    s_idx = p_idx.sentence_by_index(idx)
    print("Sentence by index: ", s_idx.name)

    s_idx = train.sentences_by_phone_df(phn).sentence[idx]
    print("Sentence by phone: ", s_idx.name)

    # Waveform
    s_idx_wav = s_idx.raw_audio
    sample_rate = s_idx.sample_rate
    words_df = s_idx.words_df
    phones_df = s_idx.phones_df
    word_counts = s_idx.word_counts
    phone_counts = s_idx.phone_counts
    
    features = au.audio_features(s_idx_wav, sample_rate)
    print(features.shape)

    # Audio feature extraction
    # Turn it into features of all sorts: 
    # [mel energies + mel filterbank + log mel filterbank + mfcc signal]
    print(s_idx_wav.shape);assert(0)
    gained_padded_audio = au.audio_gained(au.audio_zero_padded(8000, s_idx_wav, 8000), 1.0)
    audio_features = au.audio_features(gained_padded_audio, sample_rate)
    sampled_audio = au.resampled_audio(s_idx_wav, 
                                       sample_rate = sample_rate, 
                                       pad = 8000, 
                                       to_sample_rate = 1000)
    print(gained_padded_audio.shape, sampled_audio.shape, audio_features.shape)

    # Get down-sampled words and phoneme sequences
    sentence_phones_input=au.resampled_phones_df(s_idx.phones_df,s_idx.sample_rate,left_pad=8000)
    sentence_words_input = au.resampled_phones_df(s_idx.words_df,s_idx.sample_rate,left_pad=8000)

    
    # Draw everything
    #   1. Raw audio
    #   2. Raw word transcriptions
    #   3. Raw phoneme transcriptions
    #   4. Downsampled audio
    #   5. Downsampled word transcriptions
    #   6. Downsampled phoneme transcriptions
    #   7. Mel features

    # 1. Raw audio
    du.DrawVerticalPanels([du.AudioPanel(s_idx.raw_audio, show_x_axis=True)], scale=scale_wav)
    if args.save: plt.savefig(args.out_path + "/waveform_phn_"+phn+"_idx_"+str(idx) +".png")
        
    # 2. Raw word transcriptions
    du.DrawVerticalPanels([ du.WordsPanel(s_idx.words_df, s_idx.raw_audio.shape[0],
                                                          show_x_axis=True)], scale=scale_words)
    if args.save: plt.savefig(args.out_path + "/words_phn_"+ phn + "_idx_"+str(idx)+".png")
        
    # 3. Raw phoneme transcriptions
    du.DrawVerticalPanels([du.PhonesPanel(s_idx.phones_df, s_idx.raw_audio.shape[0])], 
                                                                     scale=scale_phones)
    if args.save: plt.savefig(args.out_path + "/phones_phn_"+ phn + "_idx_"+str(idx)+".png")
        
    # 4. Downsampled audio
    du.DrawVerticalPanels([du.AudioPanel(sampled_audio, show_x_axis=True)], scale=scale_wav)
    if args.save: plt.savefig(args.out_path + "/sampledWav_phn_"+phn+"_idx_"+str(idx)+".png")
        
    # 5. Downsampled word transcriptions
    du.DrawVerticalPanels([du.WordsPanel(sentence_words_input, sampled_audio.shape[0], 
                                                     show_x_axis=True)], scale=scale_words)   
    if args.save: plt.savefig(args.out_path + "/sampled_words_phn_"+phn+"_idx_"+str(idx)+".png")
        
    # 6. Downsampled phoneme transcriptions
    du.DrawVerticalPanels([du.PhonesPanel(sentence_phones_input, sampled_audio.shape[0])], 
                                                                         scale=scale_phones)
    if args.save: plt.savefig(args.out_path+"/sampled_phones_phn_"+phn+"_idx_"+str(idx)+".png")
        
    # 7. Mel features
    du.DrawVerticalPanels([du.SignalsPanel(audio_features)], scale=scale_wav/2)
    if args.save: plt.savefig(args.out_path + "/Mel_features_phn_"+phn+"_idx_"+str(idx)+".png")
        
    if args.show: plt.show()

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='timit_preprocess',description="Visualize timit data ")
    parser.add_argument("--data_path",type=str,default='timit/data',help="Directo of Timit data")
    parser.add_argument("--scale_wav",type=float,default=1.0,help="scale of waveform")
    parser.add_argument("--scale_words",type=float,default=1.5,help="scale for words visualize")
    parser.add_argument("--scale_phones",type=float,default=1.5,help="scale  phones visualize")
    parser.add_argument("--phn",type=str,default='aa',help="choose phn to visualize .")
    parser.add_argument("--idx",type=int,default=1,help="choose a sentence .")
    parser.add_argument("--save",type=bool,default=True,help="save plots.")
    parser.add_argument("--out_path",type=str,default="plots",help="path to save plots.")
    parser.add_argument("--show",type=bool,default=False,help="show plots.")
        
    args = parser.parse_args()

    #Visualize(args)
    Visualize_features(args)
