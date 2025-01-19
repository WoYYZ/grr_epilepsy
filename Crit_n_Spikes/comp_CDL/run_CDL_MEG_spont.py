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

def butter_lowpass(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    y = filtfilt(b, a, data)
    return y

def butter_highpass(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    y = filtfilt(b, a, data)
    return y
    
def get_basename(filepath):
    return os.path.basename(filepath).split('.')[0]


def main(args):
    
    ori_fs             = 1000 # Original sampling rate (Hz)
    decimate           = 10   # dec factor for analysis data
    HI_PASS            = 1    #Hz 
    LOW_PASS           = 30   #Hz
    fs                 = ori_fs // decimate  # decimated signal fs
    n_atoms            = 30
    ATOM_DURATION      = 3    # seconds
    SPLIT_BLOCKS       = 8    # Spliting a long resting into x blocks
    N_JOBS             = 80   # cdl jobs


    # data in and out
    SUBJ_DIR  = Path('/m/nbe/scratch/grr_epilepsy/MEG_spont')
    SUBJS     = [file for file in os.listdir(SUBJ_DIR) if file.endswith('.npy')]
    SUBJS     = sorted(SUBJS)
    
    MAT_FILE_WRITEOUT  = Path('/m/nbe/scratch/grr_epilepsy/MEG_spont/results/')
    MAT_FILE_WRITEOUT  = MAT_FILE_WRITEOUT / 'WHOLE_COHORT_CDL_from_Triton'



    t1 = time.time()
    # for all subjects ##############################################################################
    for subj in SUBJS:    
    
        t11=time.time()
        filePath = Path(SUBJ_DIR / subj)
        parcelTS = np.load(filePath) # data: channels x samples    
        PARC_N       = parcelTS.shape[0]
        SAMPLE_N     = signal.decimate(parcelTS[0], decimate).shape[0]
        bpTS         = np.zeros([PARC_N, SAMPLE_N])
    
        # band-pass the data, and decimate if need be #######################
        for i in np.arange(0,PARC_N):
            aParcelTS            = parcelTS[i]    
            lowpass_data         = butter_lowpass(aParcelTS, LOW_PASS, ori_fs)
            highpass             = butter_highpass(lowpass_data, HI_PASS, ori_fs)
            bpTS[i, :]           = signal.decimate(highpass, decimate)
        del lowpass_data
        print('Band-pass done, now fitting: '+subj+ '...')
    
        # Split a long trial into x blocks
        split_parcTS = split_signal(bpTS[None], SPLIT_BLOCKS)
    
        # Define the shape of the dictionary
        n_times_atom = int(round(fs * ATOM_DURATION)) 
        print('Data shape:', split_parcTS.shape, '; N of Atoms: ',  n_atoms, '; Sample per atom: ', n_times_atom)
    
        cdl = BatchCDL(
            # Shape of the dictionary
            n_atoms=n_atoms,
            n_times_atom=n_times_atom,
            # Request a rank1 dictionary with unit norm temporal and spatial maps
            rank1=True, uv_constraint='separate',
            # Initialize the dictionary with random chunk from the data
            D_init='chunk',
            # rescale the regularization parameter to be 20% of lambda_max
            lmbd_max="scaled", reg=.2,
            # Number of iteration for the alternate minimization and cvg threshold
            n_iter=100, eps=1e-4,
            # solver for the z-step
            solver_z="lgcd", solver_z_kwargs={'tol': 1e-2, 'max_iter': 1000},
            # solver for the d-step
            solver_d='alternate_adaptive', solver_d_kwargs={'max_iter': 300},
            # Technical parameters
            verbose=1, random_state=0, n_jobs=N_JOBS)
        
    
        ######################################### FIT IT   
        cdl.fit(split_parcTS)
        del split_parcTS   
        ######################################### FIT IT   
    
    
        u_hat = cdl.u_hat_ # spatial
        v_hat = cdl.v_hat_ # temporal
        z_hat = cdl.z_hat_ # scores
        #print(v_hat.shape)
    
        # save results
        writeoutPrefix =subj[0:13] + '_seg00_'
        matfile = MAT_FILE_WRITEOUT / (writeoutPrefix + '.mat')    
    
        FILENAME = (writeoutPrefix + '_BB_' + str(HI_PASS) + 
                    '-' + str(LOW_PASS) + 'Hz_' + str(n_atoms) + 
                    '_Atoms_' + str(ATOM_DURATION) + '(s).mat')
        MAT_FILE = MAT_FILE_WRITEOUT / FILENAME
    
        t12=time.time()
        print('\tTime spent:', round((t12-t11)/60,1), ' mins.' )
        timeSpent_min = round((t12-t11)/60,1)
        savemat(MAT_FILE, {"u_hat": u_hat, "v_hat": v_hat, 'z_hat': z_hat, 'fs': fs, 'timeSpent_min':timeSpent_min})
        print('Writing: ' + FILENAME)    
    
    
    t2 = time.time()    
    print('\nN of atoms: ', n_atoms, ', Num of Jobs: ', N_JOBS)
    print('Low: ', LOW_PASS, 'Hz, High:', HI_PASS, 'Hz')
    print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t2))}")
    print('Total time spent:', round((t2-t1)/60,1), ' mins.' ) 





    
    
    #save_data = {'wpli': wpli_as_freq, 'cplv': cplv_as_freq, 'dfa': dfa_as_freq, 'fei': fei_as_freq}
    #basename = get_basename(args.filepath)
    #save_path = os.path.join('/scratch/nbe/grr_epilepsy/observables/', f'{basename}_observables.pickle')
    #pickle.dump(save_data, open(save_path, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-filepath', '--filepath')
    args = parser.parse_args()
    
    main(args)
    
