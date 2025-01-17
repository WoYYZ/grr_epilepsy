import os

import numpy as np
import cupy as cp
import scipy as sp

import pickle
import sys
import argparse

import crosspy

def load_data(filepath):
    if filepath.endswith('npy'):
        res = np.load(filepath)
    else:
        data_obj = sp.io.loadmat(filepath)
        res = data_obj['data']

    return res.astype(np.double)

def get_basename(filepath):
    return os.path.basename(filepath).split('.')[0]

def main(args):
    n_parcels = 200
    sampling_freq = 1000
    omega = 7.5
    min_freq = 1
    max_freq = 225

    frequencies_of_interest = np.geomspace(min_freq, max_freq, 50)

    dfa_min_win = 20
    dfa_max_fraction = 10

    fei_win_multiplier = 50
    
    data_broadband = load_data(args.filepath)

    # cPLV, wPLI, DFA, fE/I 
    wpli_as_freq = np.zeros((len(frequencies_of_interest), n_parcels, n_parcels))
    cplv_as_freq = np.zeros((len(frequencies_of_interest), n_parcels, n_parcels), dtype=complex)
    dfa_as_freq = np.zeros((len(frequencies_of_interest), n_parcels))
    fei_as_freq = np.zeros((len(frequencies_of_interest), n_parcels))

    for freq_idx, frequency in enumerate(frequencies_of_interest):
        print(f'Processing frequency {frequency}Hz')
        
        dfa_window_sizes = np.geomspace(dfa_min_win*sampling_freq/frequency, data_broadband.shape[-1]//dfa_max_fraction, 30).astype(int)
        fei_window_size = int(fei_win_multiplier*sampling_freq/frequency)
        
        data_filt = crosspy.preprocessing.signal.filter_data(data_broadband, sfreq=sampling_freq, frequency=frequency, omega=omega, n_jobs='cuda')
        data_envelope = np.abs(data_filt)
    
        wpli_as_freq[freq_idx] = crosspy.core.synchrony.wpli(data_filt).get()
        cplv_as_freq[freq_idx] = crosspy.core.synchrony.cplv(data_filt).get()
        dfa_as_freq[freq_idx] = crosspy.core.criticality.dfa(data_envelope, dfa_window_sizes)[2]
        fei_as_freq[freq_idx] = crosspy.core.criticality.fei(data_envelope.get(), fei_window_size, force_gpu=False) # some weird error so we cant use GPU for fEI???
        # fei_as_freq[freq_idx] = crosspy.core.criticality.fei(data_envelope, fei_window_size) # some weird error???
    
    save_data = {'wpli': wpli_as_freq, 'cplv': cplv_as_freq, 'dfa': dfa_as_freq, 'fei': fei_as_freq}

    basename = get_basename(args.filepath)
    save_path = os.path.join('/scratch/nbe/grr_epilepsy/observables/', f'{basename}_observables.pickle')
    pickle.dump(save_data, open(save_path, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-filepath', '--filepath')
    args = parser.parse_args()
    
    main(args)
    
