#%%
import pickle, gzip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob, os
import re, ast
import itertools
import seaborn as sns
from matplotlib import rcParams
from qiskit.visualization import plot_histogram
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from seaborn import set_theme
from matplotlib import style
from multiprocessing import Pool, cpu_count
from os import scandir

rcParams.update({'figure.autolayout': True})
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['text.usetex'] = True

#%%

def compute_QVF_michelson_contrast_single_injection(df, circuit_name, phi, theta):
    """Compute the new QVF for the whole circuit, as well as for each available qubit"""
    dfFilter = df[(df.circuit_name==circuit_name) & (df.first_phi==phi) & (df.first_theta==theta)]
    QVF = {}
    QVF['QVF_circuit'] = dfFilter['QVF'].mean()
    
    qubits = set(dfFilter['first_qubit_injected'])    
    for q in qubits:
        QVF['QVF_qubit_'+str(q)] = dfFilter[dfFilter.first_qubit_injected==q]['QVF'].mean()
    return QVF
    

def compute_QVF_michelson_contrast_double_injection(df, circuit_name, phi_0, theta_0, phi_1, theta_1):
    """Compute the new QVF for the whole circuit, as well as for each available qubit"""
    dfFilter = df[(df.circuit_name == circuit_name) & (df.first_phi == phi_0) & (df.first_theta == theta_0)& (df.second_phi == phi_1) & (df.second_theta == theta_1)]
    QVF = {}
    QVF['QVF_circuit'] = dfFilter['QVF'].mean()

    qubits = set(dfFilter['first_qubit_injected'])
    for q in qubits:
        QVF['QVF_qubit_' +
            str(q)] = dfFilter[dfFilter.first_qubit_injected == q]['QVF'].mean()
    return QVF

def QVF_michelson_contrast(gold_bitstring, answer, shots):
    """Compute Michelson contrast between gold and highest percentage fault string"""
    # Sort the answer, position 0 has the highest bitstring, position 1 the second highest
    answer_sorted = sorted(answer, key=answer.get, reverse=True)
    
    # If gold bitstring is not in answer, percentage is zero
    if gold_bitstring not in answer:
        good_percent = 0
    else:
        good_percent = answer[gold_bitstring]/shots
        
    if answer_sorted[0] == gold_bitstring: # gold bitstring has the highest count (max)
        # next bitstring is the second highest
        next_percent = answer[answer_sorted[1]]/shots 
        next_bitstring = answer_sorted[1]
    else: # gold bitstring has NOT the highest count (not max)
        next_percent = answer[answer_sorted[0]]/shots 
        next_bitstring = answer_sorted[0]
    qvf = (good_percent - next_percent) / (good_percent + next_percent)    
    return 1 - (qvf+1)/2, next_bitstring
    
def build_DF_newQVF(data):
    """Read pickled data and store results in a dataframe"""
    results = []
    shots = 1024
    gold_bitstring = max(data['output_gold_noise'], key=data['output_gold_noise'].get)#check
    original_gold_percentage = data['output_gold_noise'][gold_bitstring]/shots

    for i, answer in enumerate(data['output_injections_noise']):
        qvf, next_bitstring = QVF_michelson_contrast(gold_bitstring, answer, shots)
        max_key = max(answer, key=answer.get)
        output_percentage = answer[max_key]/shots
        next_bitstring_percentage = answer[next_bitstring]/shots
        if gold_bitstring not in answer:
            gold_percentage = 0
        else:
            gold_percentage = answer[gold_bitstring]/shots
        result = {'gold_bitstring':gold_bitstring
                , 'gold_count_percentage':gold_percentage
                , 'original_gold_count_percentage':original_gold_percentage
                , 'next_bitstring': next_bitstring
                , 'next_bitstring_percentage': next_bitstring_percentage
                , 'QVF':qvf
                , 'first_qubit_injected':data['wires'][i]
                , 'first_phi':data['phi0']
                , 'first_theta':data['theta0']
                , 'second_qubit_injected':data['second_wires'][i]
                , 'second_phi':data['phi1']
                , 'second_theta':data['theta1']
                #, 'gate_injected':data['circuits_injections'][i].metadata['gate_inserted']
                #, 'lambda':data['circuits_injections'][i].metadata['lambda']
                , 'circuit_name':data['name']
                }
        results.append(result)
    return pd.DataFrame(results)

#%%
def filter_single_fi(results):
    """Returns only the faults for which no secondary fault was injected"""
    return results[(results.second_phi == 0) & (results.second_theta == 0)]

def read_results_double_fi(filenames):
    """Process double fault injection results files and return all data"""
    # read all data and insert it into one dataframe
    df_newQVF = pd.DataFrame()
    for filename in filenames:
        data = pickle.load(gzip.open(filename, 'r'))
        
        for d in data:
            df_newQVF = pd.concat([df_newQVF, build_DF_newQVF(d)], ignore_index=True)
        del data
    
    return df_newQVF

def read_file(filename):
    df_newQVF = pd.DataFrame()
    data = pickle.load(gzip.open(filename, 'r'))
    for d in data:
        df_newQVF = pd.concat([df_newQVF, build_DF_newQVF(d)], ignore_index=True)
    del data
    return df_newQVF

def read_results_directory(dir):
    """Process double fault injection results directory and return all data"""
    filenames = []
    for filename in scandir(dir):
        if filename.is_file():
            filenames.append(filename.path)

    pool = Pool(cpu_count())
    df = pd.concat(pool.map(read_file, filenames))
    pool.close()
    pool.join()
    return df

def read_results_single_fi(filenames):
    """Process single/double fault injection results files and return only single faults data"""
    df_results = read_results_double_fi(filenames)
    df_only_single = filter_single_fi(df_results)

    return df_only_single

#%%

def get_circuits_angles(results):
    phi_list = list(set(results.first_phi))
    phi_list.sort(reverse=False)
    theta_list = list(set(results.first_theta))
    theta_list.sort()
    circuits = list(set(results.circuit_name))
    circuits.sort()

    return circuits, theta_list, phi_list

def get_processed_table(results):
    """Get available circuits and parameters used for injection (phi and theta)"""

    circuits, theta_list, phi_list = get_circuits_angles(results)

    table = []
    for circuit in circuits:
        for phi in phi_list:
            for theta in theta_list:
                qvf = compute_QVF_michelson_contrast_single_injection(results, circuit, phi, theta)
                qvf['circuit_name'] = circuit
                qvf['first_phi'] = phi
                qvf['first_theta'] = theta
                #qvf['threshold'] = threshold
                table.append(qvf)

    return table

def process_double_fi(results):
    """Process results and condensate them for each qubit and fault position in the circuit"""
    table = get_processed_table(results)

    new_qvfDF_noise = pd.DataFrame(table)
    new_qvfDF_noise = new_qvfDF_noise[sorted(list(new_qvfDF_noise.columns))]

    return new_qvfDF_noise

def process_single_fi(results):
    """Process single fault injection results and condensate them for each qubit and fault position in the circuit"""
    table = get_processed_table(results)

    new_qvfDF_noise_single = pd.DataFrame(table)
    new_qvfDF_noise_single = new_qvfDF_noise_single[sorted(list(new_qvfDF_noise_single.columns))]

    return new_qvfDF_noise_single

#%%

def compute_merged_histogram(results, savepath="./plots/histograms/"):
    """Compute and save result histograms"""

    circs = [process_single_fi(results), process_double_fi(results)]
    qvf_tmp = list()
    for i, dfCirc in enumerate(circs):
        qvf_tmp.append(dfCirc)
        qvf_tmp[i] = qvf_tmp[i].pivot('first_phi', 'first_theta', 'QVF_circuit')
        qvf_tmp[i].columns.name = '$\\theta$ shift'
        qvf_tmp[i].index.name = '$\\phi$ shift'

        all_values = []
        for column in qvf_tmp[i]:
            this_column_values = qvf_tmp[i][column].tolist()
            all_values += this_column_values
        one_column_df = pd.DataFrame(all_values)
        print('mean:',one_column_df.mean()[0],' std:',one_column_df.std()[0])

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.set(font_scale=1.3)
    colorsDist = ['black', 'red', 'green', 'blue']
    for i, data in enumerate(qvf_tmp):
        sns.distplot(data, bins=256, color=colorsDist[i])
    #sns.distplot(qvf_tmp2, bins=256, color='red')
    plt.xlim(0, 1)

    # comparing distribution of single FI vs double FI
    if not os.path.exists(savepath):
        os.makedirs(savepath) 
    tmpFileName = savepath+circs[0].circuit_name[0]+'_merged_distribution_histogram'+'.pdf'
    fig.savefig(tmpFileName, bbox_inches = 'tight')
    plt.close()

#%%

def compute_circuit_heatmaps(results, savepath="./plots/heatmaps/"):
    """Compute and save results heatmaps"""
    theta_list_tex = ['0', '', '', '$\\frac{\pi}{4}$', '', '', '$\\frac{\pi}{2}$', ''
                , '', '$\\frac{3\pi}{4}$', '', '', '$\pi$']
    phi_list_tex = ['', '', '$\\frac{7\pi}{4}$', '', ''
                , '$\\frac{6\pi}{4}$', '', '', '$\\frac{5\pi}{4}$', ''
                , '', '$\pi$', '', '',  '$\\frac{3\pi}{4}$'  
                , '', '', '$\\frac{\pi}{2}$', '', '', '$\\frac{\pi}{4}$'
                , '', '', '0']

    circs = [process_single_fi(results), process_double_fi(results)]          
    for i,circuitDF in enumerate(circs):
        qvf_tmp = circuitDF
        qvf_tmp = qvf_tmp.pivot('first_phi', 'first_theta', 'QVF_circuit')
        qvf_tmp.columns.name = '$\\theta$ shift'
        qvf_tmp.index.name = '$\\phi$ shift'
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        param={'label': 'QVF'}

        divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
        rdgn = sns.diverging_palette(h_neg=130, h_pos=10, s=200, l=55, sep=20, as_cmap=True)
        sns.set(font_scale=1.3)
        ax = sns.heatmap(qvf_tmp, xticklabels=theta_list_tex, yticklabels=phi_list_tex, cmap=rdgn, cbar_kws=param, vmin=0, vmax=1)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        if i == 0:
            fig.savefig(savepath+circs[0].circuit_name[0]+'_single_heatmap.pdf', bbox_inches='tight')
        else:
            fig.savefig(savepath+circs[0].circuit_name[0]+'_double_heatmap.pdf', bbox_inches='tight')
        plt.close()

def compute_circuit_delta_heatmaps(results, savepath="./plots/deltaHeatmaps/"):
    """Plot delta heatmaps between single and double FI"""

    theta_list_tex = ['0', '', '', '$\\frac{\pi}{4}$', '', '', '$\\frac{\pi}{2}$', ''
                , '', '$\\frac{3\pi}{4}$', '', '', '$\pi$']
    phi_list_tex = ['', '', '$\\frac{7\pi}{4}$', '', ''
                , '$\\frac{6\pi}{4}$', '', '', '$\\frac{5\pi}{4}$', ''
                , '', '$\pi$', '', '',  '$\\frac{3\pi}{4}$'  
                , '', '', '$\\frac{\pi}{2}$', '', '', '$\\frac{\pi}{4}$'
                , '', '', '0']
    
    qvf_single_fi = process_single_fi(results)
    qvf_double_fi = process_double_fi(results)

    qvf_tmp = qvf_single_fi.copy()
    qvf_tmp['delta'] = qvf_double_fi['QVF_circuit'] - qvf_single_fi['QVF_circuit']
    qvf_tmp = qvf_tmp.pivot('first_phi', 'first_theta', 'delta')
    qvf_tmp.columns.name = '$\\theta$ shift'
    qvf_tmp.index.name = '$\\phi$ shift'
    #fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    label = '$\Delta$QVF = Double - Single'
    param={'label': label}
    ax = sns.heatmap(qvf_tmp, xticklabels=theta_list_tex, yticklabels=phi_list_tex, cmap='seismic', cbar_kws=param, vmin=-1, vmax=1)            
    plt.axhline(y=9.5, color="blue", linestyle="--")       #T at phi=pi/4
    plt.text(13.25, 9.7, r'$T$', fontsize=10, color="blue")            
    plt.axhline(y=6.5, color="black", linestyle="--")      #S at phi=pi/2  
    plt.text(13.25, 6.7, r'$S$', fontsize=10, color="black")       
    plt.axhline(y=0.5, color="cyan", linestyle="--")       #Z at phi=pi   
    plt.text(13.25, 0.7, r'$Z$', fontsize=10, color="cyan")          
    plt.axvline(x=12.5, color="purple", linestyle="--")     #X,Y at theta=pi
    plt.text(12, -0.5, r'$X, Y$', fontsize=10, color="purple")
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    fig.savefig(savepath+qvf_single_fi.circuit_name[0]+'_double_single_delta_heatmap.pdf', bbox_inches='tight')
    plt.close()

#%%

def compute_qubit_histograms(results, savepath="./plots/histograms/"):
    """Compute single qubit histograms"""

    circuits, theta_list, phi_list = get_circuits_angles(results)
    new_qvfDF_noise = process_double_fi(results)

    for circuit in circuits:
        # get a list of qubit columns (circuit may have different number of qubits now)
        colNames = new_qvfDF_noise[new_qvfDF_noise['circuit_name']==circuit].dropna(axis=1).columns    
        QVF_list= ['QVF_circuit']
        QVF_list.extend( [x for x in colNames if re.search('QVF_qubit_.*',x)] ) # uncomment this line to include individual qubit analysis
        
        for qvf_idx in QVF_list:
            qvf_tmp = new_qvfDF_noise[new_qvfDF_noise['circuit_name']==circuit]
            qvf_tmp = qvf_tmp.pivot('first_phi', 'first_theta', qvf_idx)
            qvf_tmp.columns.name = '$\\theta$ shift'
            qvf_tmp.index.name = '$\\phi$ shift'
            
            all_values = []
            for column in qvf_tmp:
                this_column_values = qvf_tmp[column].tolist()
                all_values += this_column_values
            one_column_df = pd.DataFrame(all_values)

            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            sns.set(font_scale=1.3)
            ax = sns.distplot(qvf_tmp, bins=256, color='black')
            plt.xlim(0, 1)

            tmp_mean = one_column_df.mean()
            tmp_stddev = one_column_df.std()
            ax.get_yaxis().set_visible(False)
            tmpFileName = savepath+circuit+'_'+qvf_idx+'_distribution_histogram_'+str(tmp_mean[0])+'_'+str(tmp_stddev[0])+'.pdf'
            fig.savefig(tmpFileName, bbox_inches = 'tight')
            plt.close()

def compute_qubit_heatmaps(results, savepath="./plots/heatmaps/"):
    """Compute single qubit histograms"""

    theta_list_tex = ['0', '', '', '$\\frac{\pi}{4}$', '', '', '$\\frac{\pi}{2}$', ''
                    , '', '$\\frac{3\pi}{4}$', '', '', '$\pi$']
    phi_list_tex = ['', '', '$\\frac{7\pi}{4}$', '', ''
                , '$\\frac{6\pi}{4}$', '', '', '$\\frac{5\pi}{4}$', ''
                , '', '$\pi$', '', '',  '$\\frac{3\pi}{4}$'  
                , '', '', '$\\frac{\pi}{2}$', '', '', '$\\frac{\pi}{4}$'
                , '', '', '0']

    circuits, theta_list, phi_list = get_circuits_angles(results)
    new_qvfDF_noise = process_double_fi(results)

    for circuit in circuits:
        # get a list of qubit columns (circuit may have different number of qubits now)
        colNames = new_qvfDF_noise[new_qvfDF_noise['circuit_name']==circuit].dropna(axis=1).columns    
        QVF_list= ['QVF_circuit']
        QVF_list.extend( [x for x in colNames if re.search('QVF_qubit_.*',x)] ) # uncomment this line to include individual qubit analysis
        
        for qvf_idx in QVF_list:
            qvf_tmp = new_qvfDF_noise[new_qvfDF_noise['circuit_name']==circuit]
            qvf_tmp = qvf_tmp.pivot('first_phi', 'first_theta', qvf_idx)
            qvf_tmp.columns.name = '$\\theta$ shift'
            qvf_tmp.index.name = '$\\phi$ shift'
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            param={'label': 'QVF'}

            divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
            rdgn = sns.diverging_palette(h_neg=130, h_pos=10, s=200, l=55, sep=20, as_cmap=True)
            sns.set(font_scale=1.3)
            ax = sns.heatmap(qvf_tmp, xticklabels=theta_list_tex, yticklabels=phi_list_tex, cmap=rdgn, cbar_kws=param, vmin=0, vmax=1)
            fig.savefig(savepath+circuit+'_'+qvf_idx+'_heatmap.pdf', bbox_inches='tight')
            plt.close()
