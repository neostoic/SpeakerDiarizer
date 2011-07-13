import unittest
import pylab as pl
import matplotlib as mpl
import itertools
import sys
import math
import timeit
import copy
import time
import struct
import scipy.stats.mstats as stats
import ConfigParser
import os.path
import getopt

from em import *


MINVALUEFORMINUSLOG = -1000.0

def generate_synthetic_data(N):
    np.random.seed(0)
    C = np.array([[0., -0.7], [3.5, .7]])
    C1 = np.array([[-0.4, 1.7], [0.3, .7]])
    Y = np.r_[
        np.dot(np.random.randn(N/3, 2), C1),
        np.dot(np.random.randn(N/3, 2), C),
        np.random.randn(N/3, 2) + np.array([3, 3]),
        ]
    return Y.astype(np.float32)

class EMTester(object):

    def __init__(self, from_file, f_file_name, sp_file_name, variant_param_space, device_id):
        self.results = {}
        self.variant_param_space = variant_param_space
        self.device_id = device_id
        if from_file:

            f = open(f_file_name, "rb")
            sp = open(sp_file_name, "r")

            print "Reading in HTK feature file..."

            #=== Read Feature File ==
            try:
                nSamples = struct.unpack('>i', f.read(4))[0]
                sampPeriod = struct.unpack('>i', f.read(4))[0]
                sampSize = struct.unpack('>h', f.read(2))[0]
                sampKind = struct.unpack('>h', f.read(2))[0]

                print "Total number of frames read: ", nSamples
                self.total_num_frames = nSamples
                
                D = sampSize/4 #dimension of feature vector
                l = []
                count = 0
                while count < (nSamples * D):
                    bFloat = f.read(4)
                    fl = struct.unpack('>f', bFloat)[0]
                    l.append(fl)
                    count = count + 1
            finally:
                f.close()

            #=== Prune to Speech Only ==
            print "Reading in speech/nonspeech file..."
            
            l_start = []
            l_end = []
            num_speech_frames = 0
            for line in sp:
                s = line.split(' ')
                st = math.floor(100 * float(s[2]) + 0.5)
                en = math.floor(100 * float(s[3].replace('\n','')) + 0.5)
                st1 = int(st)
                en1 = int(en)
                l_start.append(st1*19)
                l_end.append(en1*19)
                num_speech_frames = num_speech_frames + (en1 - st1 + 1)

            print "Total number of speech frames: ", num_speech_frames
            pruned_list = []
            total = 0
            for start in l_start:
                end = l_end[l_start.index(start)]
                total += (end/19 - start/19 + 1)
                x = 0
                index = start
                while x < (end-start+19):
                    pruned_list.append(l[index])
                    index += 1
                    x += 1

            floatArray = np.array(pruned_list, dtype = np.float32)
            self.X = floatArray.reshape(num_speech_frames, D)
            
            self.N = self.X.shape[0]
            self.D = self.X.shape[1]
        else:
            N = 1000
            self.X = generate_synthetic_data(N)
            self.N = self.X.shape[0]
            self.D = self.X.shape[1]

    def write_to_RTTM(self, rttm_file_name, sp_file_name, meeting_name, most_likely, num_gmms):

        print "Writing out RTTM file..."

        #do majority voting in chunks of 250
        duration = 250
        chunk = 0
        end_chunk = duration

        max_gmm_list = []

        smoothed_most_likely = np.array([], dtype=np.float32)

        while end_chunk < len(most_likely):
            chunk_arr = most_likely[range(chunk, end_chunk)]
            max_gmm = stats.mode(chunk_arr)[0][0]
            max_gmm_list.append(max_gmm)
            smoothed_most_likely = np.append(smoothed_most_likely, max_gmm*np.ones(250))
            chunk += duration
            end_chunk += duration

        end_chunk -= duration
        if end_chunk < len(most_likely):
            chunk_arr = most_likely[range(end_chunk, len(most_likely))]
            max_gmm = stats.mode(chunk_arr)[0][0]
            max_gmm_list.append(max_gmm)
            smoothed_most_likely = np.append(smoothed_most_likely, max_gmm*np.ones(len(most_likely)-end_chunk))

        most_likely = smoothed_most_likely
        
        out_file = open(rttm_file_name, 'w')

        with_non_speech = -1*np.ones(self.total_num_frames)
        
        speech_seg = np.loadtxt(sp_file_name, delimiter=' ',usecols=(2,3))
        speech_seg_i = np.round(speech_seg*100).astype('int32')
        sizes = np.diff(speech_seg_i)
        
        sizes = sizes.reshape(sizes.size)
        offsets = np.cumsum(sizes)
        offsets = np.hstack((0, offsets[0:-1]))

        offsets += np.array(range(len(offsets)))
        
        #populate the array with speech clusters
        speech_index = 0
        counter = 0
        for pair in speech_seg_i:
            st = pair[0]
            en = pair[1]
            speech_index = offsets[counter]

            counter+=1
            idx = 0
            for x in range(st+1, en+1):
                with_non_speech[x] = most_likely[speech_index+idx]
                idx += 1

        cnum = with_non_speech[0]
        cst  = 0
        cen  = 0
        for i in range(1,self.total_num_frames): 
            if with_non_speech[i] != cnum: 
                if (cnum >= 0):
                    start_secs = ((cst)*0.01)
                    dur_secs = (cen - cst + 2)*0.01
                    out_file.write("SPEAKER " + meeting_name + " 1 " + str(start_secs) + " "+ str(dur_secs) + " <NA> <NA> " + "speaker_" + str(cnum) + " <NA>\n")

                    
                cst = i
                cen = i
                cnum = with_non_speech[i]
            else:
                cen+=1
                  
        if cst < cen:
            cnum = with_non_speech[self.total_num_frames-1]
            if(cnum >= 0):
                start_secs = ((cst+1)*0.01)
                dur_secs = (cen - cst + 1)*0.01
                out_file.write("SPEAKER " + meeting_name + " 1 " + str(start_secs) + " "+ str(dur_secs) + " <NA> <NA> " + "speaker_" + str(cnum) + " <NA>\n")


        print "DONE"
        
    def new_gmm(self, M):
        self.M = M
        self.gmm = GMM(self.M, self.D, self.variant_param_space, self.device_id)

    def new_gmm_list(self, M, k):
        self.M = M
        self.init_num_clusters = k
        self.gmm_list = [GMM(self.M, self.D, self.variant_param_space, self.device_id) for i in range(k)]


    def segment_majority_vote(self):
        
        num_clusters = len(self.gmm_list)

        # Resegment data based on likelihood scoring
        likelihoods = self.gmm_list[0].score(self.X)
        for g in self.gmm_list[1:]:
            likelihoods = np.column_stack((likelihoods, g.score(self.X)))
        most_likely = likelihoods.argmax(axis=1)

        # Across 2.5 secs of observations, vote on which cluster they should be associated with

        iter_training = {}
        interval_size = 250

        for i in range(interval_size, self.N, interval_size):
            arr = np.array(most_likely[(range(i-interval_size, i))])
            max_gmm = int(stats.mode(arr)[0][0])
            iter_training.setdefault((self.gmm_list[max_gmm],max_gmm),[]).append(self.X[i-interval_size:i,:])
        
        arr = np.array(most_likely[(range((self.N/interval_size)*interval_size, self.N))])
        max_gmm = int(stats.mode(arr)[0][0])
        iter_training.setdefault((self.gmm_list[max_gmm], max_gmm),[]).append(self.X[(self.N/interval_size)*interval_size:self.N,:])
                
        iter_bic_dict = {}
        iter_bic_list = []
        cluster_count = 0
        for gp, data_list in iter_training.iteritems():
            g = gp[0]
            p = gp[1]
            cluster_data =  data_list[0]
            for d in data_list[1:]:
                cluster_data = np.concatenate((cluster_data, d))
            cluster_data = np.ascontiguousarray(cluster_data)
            
            g.train(cluster_data)
            iter_bic_list.append((g,cluster_data))
            iter_bic_dict[p] = cluster_data
            cluster_count += 1

        return iter_bic_dict, iter_bic_list, most_likely
        
    def cluster(self, KL_ntop, NUM_SEG_LOOPS_INIT, NUM_SEG_LOOPS):
        
        main_start = time.time()

        # ----------- Uniform Initialization -----------
        # Get the events, divide them into an initial k clusters and train each GMM on a cluster
        per_cluster = self.N/self.init_num_clusters
        init_training = zip(self.gmm_list,np.vsplit(self.X, range(per_cluster, self.N, per_cluster)))

        for g, x in init_training:
            g.train(x)

        # ----------- First majority vote segmentation loop ---------
        for segment_iter in range(0,NUM_SEG_LOOPS_INIT):
            iter_bic_dict, iter_bic_list, most_likely = self.segment_majority_vote()


        # ----------- Main Clustering Loop using BIC ------------

        # Perform hierarchical agglomeration based on BIC scores
        best_BIC_score = 1.0
        total_events = 0
        total_loops = 0

        while (best_BIC_score > 0 and len(self.gmm_list) > 1):

            for segment_iter in range(0,NUM_SEG_LOOPS):
                iter_bic_dict, iter_bic_list, most_likely = self.segment_majority_vote()

            # Score all pairs of GMMs using BIC
            best_merged_gmm = None
            best_BIC_score = 0.0
            merged_tuple = None
            merged_tuple_indices = None

            # ------- KL distance to compute best pairs to merge -------
            if KL_ntop > 0:
            
                top_K_gmm_pairs = self.gmm_list[0].find_top_KL_pairs(KL_ntop, self.gmm_list)
                
                for pair in top_K_gmm_pairs:
                    score = 0.0
                    gmm1idx = pair[0]
                    gmm2idx = pair[1]
                    g1 = self.gmm_list[gmm1idx]
                    g2 = self.gmm_list[gmm2idx]

                    if gmm1idx in iter_bic_dict and gmm2idx in iter_bic_dict:
                        d1 = iter_bic_dict[gmm1idx]
                        d2 = iter_bic_dict[gmm2idx]
                        data = np.concatenate((d1,d2))
                    elif gmm1idx in iter_bic_dict:
                        data = d1
                    else:
                        data = d2

                    new_gmm, score, tt = compute_distance_BIC(g1, g2, data)
                    
                    #print "Comparing BIC %d with %d: %f" % (gmm1idx, gmm2idx, score)
                    if score > best_BIC_score: 
                        best_merged_gmm = new_gmm
                        merged_tuple = (g1, g2)
                        merged_tuple_indices = (gmm1idx, gmm2idx)
                        best_BIC_score = score

            # ------- All-to-all comparison of gmms to merge -------
            else: 
                l = len(iter_bic_list)
                for gmm1idx in range(l):
                    for gmm2idx in range(gmm1idx+1, l):
                        score = 0.0
                        g1, d1 = iter_bic_list[gmm1idx]
                        g2, d2 = iter_bic_list[gmm2idx] 

                        data = np.concatenate((d1,d2))
                        new_gmm, score, tt = compute_distance_BIC(g1, g2, data)
                                                
                        #print "Comparing BIC %d with %d: %f" % (gmm1idx, gmm2idx, score)
                        if score > best_BIC_score: 
                            best_merged_gmm = new_gmm
                            merged_tuple = (g1, g2)
                            merged_tuple_indices = (gmm1idx, gmm2idx)
                            best_BIC_score = score

            # Merge the winning candidate pair if its deriable to do so

            merge_time = time.time()
            if best_BIC_score > 0.0:
                self.gmm_list.remove(merged_tuple[0])
                self.gmm_list.remove(merged_tuple[1])
                self.gmm_list.append(best_merged_gmm)
                
            print " size of each cluster:", [ g.M for g in self.gmm_list]

        print "=== Total clustering time: ", time.time()-main_start
        print "=== Final size of each cluster:", [ g.M for g in self.gmm_list]

        return most_likely

def print_usage():
    print """    ---------------------------------------------
    Speaker Diarization in Python with ASP usage:
    ---------------------------------------------
    Arguments for the diarizer are parsed from a config file. 
    Default config file is diarizer.cfg, but you can pass your own file with the '-c' option. 
    Required is the config file header: [Diarizer] and the options are as follows:
    
    --- Required: ---
    basename: \t Basename of the file to process
    mfcc_feats: \t MFCC input feature file
    output_cluster: \t Output clustering file
    M_mfcc: \t Amount of gaussains per model for mfcc
    initial_clusters: Number of initial clusters"""
    
    
    
def print_no_config():

    print "Please supply a config file with -c 'config_file_name.cfg' "
    return

def get_config_params(config):
        #read in filenames
    try:
        meeting_name = config.get('Diarizer', 'basename')
    except:
        print "basename not specified in config file! exiting..."
        sys.exit(2)
    try:
        f = config.get('Diarizer', 'mfcc_feats')
    except:
        print "Feature file mfcc_feats not specified in config file! exiting..."
        sys.exit(2)

    try:
        sp = config.get('Diarizer', 'spnsp_file')
    except:
        print "Speech file spnsp_file not specified, continuing without it..."
        sp = False

    try:
        outfile = config.get('Diarizer', 'output_cluster')
    except:
        print "output_cluster file not specified in config file! exiting..."
        sys.exit(2)
        
    #read GMM paramters
    try:
        num_gmms = int(config.get('Diarizer', 'initial_clusters'))
    except:
        print "initial_clusters not specified in config file! exiting..."
        sys.exit(2)

    try:
        num_comps = int(config.get('Diarizer', 'M_mfcc'))
    except:
        print "M_mfcc not specified in config file! exiting..."
        sys.exit(2)
        
    #read algorithm configuration
    try:
        kl_ntop = int(config.get('Diarizer', 'KL_ntop'))
    except:
        kl_ntop = 0
    try:
        num_seg_iters_init = int(config.get('Diarizer', 'num_seg_iters_init'))
    except:
        num_seg_iters_init = 2
        
    try:
        num_seg_iters = int(config.get('Diarizer', 'num_seg_iters'))
    except:
        num_seg_iters = 3

    try:
        num_em_iters = config.get('Diarizer', 'em_iterations')
    except:
        num_em_iters = 3

        
    return meeting_name, f, sp, outfile, num_gmms, num_comps, num_em_iters, kl_ntop, num_seg_iters_init, num_seg_iters



if __name__ == '__main__':
    device_id = 0
    
    #----- Main Clustering Script ----

    # Process commandline arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:", ["help"])
    except getopt.GetoptError, err:
        print_no_config()
        sys.exit(2)

    config_file = 'diarizer.cfg'
    config_specified = False
    for o, a in opts:
        if o == '-c':
            config_file = a
            config_specified = True
        if o == '--help':
            print_usage()
            sys.exit(2)
    

    if not config_specified:
        print "No config file specified, using defaul 'diarizer.cfg' file"
    else:
        print "Using the config file specified: '", config_file, "'"

    try:
        open(config_file)
    except IOError, err:
        print "Error! Config file: '", config_file, "' does not exist"
        sys.exit(2)
        
    # Parse config file
    config = ConfigParser.ConfigParser()

    config.read(config_file)

    meeting_name, f, sp, outfile, num_gmms, num_comps, num_em_iters, kl_ntop, num_seg_iters_init, num_seg_iters = get_config_params(config)
    
    variant_param_space = {
            'num_blocks_estep': ['16'],
            'num_threads_estep': ['512'],
            'num_threads_mstep': ['256'],
            'num_event_blocks': ['128'],
            'max_num_dimensions': ['50'],
            'max_num_components': ['128'],
            'max_num_dimensions_covar_v3': ['40'],
            'max_num_components_covar_v3': ['82'],
            'diag_only': ['1'],
            'max_iters': [num_em_iters],
            'min_iters': ['1'],
            'covar_version_name': ['V1']
            #'covar_version_name': ['V1', 'V2A', 'V2B', 'V3']
    }
        
    emt = EMTester(True, f, sp, variant_param_space, device_id)
    emt.new_gmm_list(num_comps, num_gmms)
    most_likely = emt.cluster(kl_ntop, num_seg_iters_init, num_seg_iters)
    emt.write_to_RTTM(outfile, sp, meeting_name, most_likely, num_gmms)


