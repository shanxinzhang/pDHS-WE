# -*- coding: utf-8 -*-
# author: longqiang luo

import os
import numpy as np
from numpy import array
from itertools import combinations, combinations_with_replacement, permutations
from repDNA.nac import RevcKmer, Kmer
from repDNA.psenac import PCPseDNC, PCPseTNC, SCPseDNC, SCPseTNC
from repDNA.util import normalize_index, get_data
import multiprocessing
import time
from util import read_k
from repDNA.ac import DAC
import random
import sys

from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.externals import joblib

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from pandas import DataFrame

def GetSequences_1(f):
    seqslst = []
    while True:
        s = f.readline()
        if not s:
            break
        else:
            if '>' not in s:
                seq = s.split('\n')[0]
                seqslst.append(seq)
    return seqslst

def GetSequences(f, alphabet):
    # return get_data(f,alphabet=alphabet)
    return get_data(f)


def RandomSampling(dataMat,number):
    try:
         slice = random.sample(dataMat, number)
         return slice
    except:
         print 'sample larger than population'


def Sampling_2(f, series):
    X = []
    i = 0
    while True:
        s = f.readline()
        if not s:
            break
        else:
            if '>' in s:
                i = i + 1
        if i in series:
            X.append(s)
            while True:
                s = f.readline()
                if not s:
                    break
                elif '>' in s:
                    i = i + 1
                    if i in series:
                        X.append(s)
                    break
                else:
                    X.append(s)
    return X


def SeriesPartion(filename, TotalLen, TrainLen, TestLen, PN):
    # f: Original File
    # TotalLen: the Sample Amount of Original File
    # TrainLen: the Amount of Train Sample
    # TestLen: the Amount of Test Sample
    # PN: 'Posis' or 'Negas'

    f = open(filename, 'r')
    sequence = range(1, TotalLen)
    seq = RandomSampling(sequence, TestLen)
    series = sorted(seq)
    series_train = series[0: TrainLen]
    series_test = series[TrainLen: TestLen]
    np.savetxt(path_train + 'Series' + PN + '.txt', series_train)
    np.savetxt(path_test + 'Series' + PN + '.txt', series_test)
    sample_train = Sampling_2(f, series_train)
    f1 = open(path_train + PN + 'AfterSampling.txt', 'w+')
    for i in sample_train:
        f1.write(i)
    f1.close()
    f.close()
    f = open(filename, 'r')
    sample_test = Sampling_2(f, series_test)
    f2 = open(path_test + PN + 'AfterSampling.txt', 'w+')
    for j in sample_test:
        f2.write(j)
    f2.close()
    f.close()
    return sample_train, sample_test


def GetKmerDict(alphabet, k):
    kmerlst = []
    partkmers = list(combinations_with_replacement(alphabet, k))
    for element in partkmers:
        elelst = set(permutations(element, k))
        strlst = [''.join(ele) for ele in elelst]
        kmerlst += strlst
    kmerlst = np.sort(kmerlst)
    kmerdict = {kmerlst[i]: i for i in range(len(kmerlst))}
    return kmerdict


############################### Spectrum Profile ##############################
def GetSpectrumProfile(k,samples_file):
    kmer = Kmer(k=k,normalize=True)
    X = kmer.make_kmer_vec(open(samples_file))
    return X
############################### Mismatch Profile ##############################
def GetMismatchProfile(instances, alphabet, k, m):
    kmerdict = GetKmerDict(alphabet, k)
    X = []
    for sequence in instances:
        vector = GetMismatchProfileVector(sequence, alphabet, kmerdict, k)
        X.append(vector)
    X = array(X)
    return X

def GetMismatchProfileVector(sequence, alphabet, kmerdict, k):
    vector = np.zeros((1, len(kmerdict)))
    n = len(sequence)
    for i in range(n - k + 1):
        subsequence = sequence[i:i + k]
        position = kmerdict.get(subsequence)
        vector[0, position] += 1
        for j in range(k):
            substitution = subsequence
            for letter in set(alphabet) ^ set(subsequence[j]):
                substitution = list(substitution)
                substitution[j] = letter
                substitution = ''.join(substitution)
                position = kmerdict.get(substitution)
                vector[0, position] += 1
    return list(vector[0])
########################### Reverse Compliment Kmer ###########################
def GetRevcKmer(k):
    rev_kmer = RevcKmer(k=k)
    pos_vec = rev_kmer.make_revckmer_vec(open(samples_file))
    X = array(pos_vec)
    return X
############ Parallel Correlation Pseudo Dinucleotide Composition #############
def GetPCPseDNC(lamada, w):
    pc_psednc = PCPseDNC(lamada = lamada, w = w)
    phyche_index = user_indices_2
    pos_vec = pc_psednc.make_pcpsednc_vec(open(samples_file),extra_phyche_index=normalize_index(phyche_index,is_convert_dict=True))
    X = array(pos_vec)
    return X
############ Parallel Correlation Pseudo Trinucleotide Composition ############
def GetPCPseTNC(lamada, w):
    pc_psetnc = PCPseTNC(lamada = lamada, w = w)
    phyche_index = user_indices_3
    pos_vec = pc_psetnc.make_pcpsetnc_vec(open(samples_file),extra_phyche_index=normalize_index(phyche_index, is_convert_dict=True))
    X = array(pos_vec)
    return X
############## Series Correlation Pseudo Dinucleotide Composition #############
def GetSCPseDNC(lamada, w):
    sc_psednc = SCPseDNC(lamada = lamada, w = w)
    phyche_index = user_indices_2
    pos_vec = sc_psednc.make_scpsednc_vec(open(samples_file), extra_phyche_index=normalize_index(phyche_index,is_convert_dict=True))
    X = array(pos_vec)
    return X
############## Series Correlation Pseudo Trinucleotide Composition ############
def GetSCPseTNC(lamada, w):
    sc_psetnc = SCPseTNC(lamada = lamada, w = w)
    phyche_index = user_indices_3
    pos_vec = sc_psetnc.make_scpsetnc_vec(open(samples_file), extra_phyche_index=normalize_index(phyche_index,is_convert_dict=True))
    X = array(pos_vec)
    return X
#####################  Dinucleotide-based auto covariance   ###############################################
def GetDAC(instances, k, lag, alphabet, extra_index_file=None, all_prop=False, theta_type=1):
    dac = DAC(lag)
    phyche_index = user_indices_2
    X = dac.make_dac_vec(open(samples_file), extra_phyche_index=normalize_index(phyche_index, is_convert_dict=True))
    return X


def GetVariousClassFeatures(samples_file, path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    fp = open(samples_file, 'r')
    sample = GetSequences(fp, 'ACGT')
    instances = array(sample)
    print('The number of samples: %d' % (len(sample)))

    # 1 Spectrum Profile for k=1,2,3,4,5
    for k in range(1, 6):
        tic = time.clock()
        X = GetSpectrumProfile(k, samples_file)
        np.savetxt(path + str(k) + '-SpectrumProfile.txt', X)
        toc = time.clock()
        print('Coding time for ' + str(k) + '-Spectrum Profile:%.3f minutes' % ((toc - tic) / 60))

    # 2 Mismatch Profile for (k,m)=(3,1),(4,1),(5,1)
    for (k, m) in [(3, 1), (4, 1), (5, 1)]:
        tic = time.clock()
        X = GetMismatchProfile(instances, alphabet, k, m)
        np.savetxt(path + str((k, m)) + '-MismatchProfile.txt', X)
        toc = time.clock()
        print('Coding time for ' + str((k, m)) + '-Mismatch Profile:%.3f minutes' % ((toc - tic) / 60))

    # 3 Reverse Compliment Kmer for k=1,2,3,4,5
    for k in range(1, 6):
        tic = time.clock()
        X = GetRevcKmer(k)
        np.savetxt(path + str(k) + '-RevcKmer.txt', X)
        toc = time.clock()
        print('Coding time for ' + str(k) + '-RevcKmer:%.3f minutes' % ((toc - tic) / 60))

    # 4 Parallel Correlation Pseudo Dinucleotide Composition
    tic = time.clock()
    X = GetPCPseDNC(3, 0.9)  #(2, 0.2)
    np.savetxt(path + 'PCPseDNC.txt', X)
    toc = time.clock()
    print('Coding time for PCPseDNC:%.3f minutes' % ((toc - tic) / 60))

    # 5 Parallel Correlation Pseudo Trinucleotide Composition
    tic = time.clock()
    X = GetPCPseTNC(3, 0.5)  #(6, 0.1)
    np.savetxt(path + 'PCPseTNC.txt', X)
    toc = time.clock()
    print('Coding time for PCPseTNC:%.3f minutes' % ((toc - tic) / 60))

    # 6 Series Correlation Pseudo Dinucleotide Composition
    tic = time.clock()
    X = GetSCPseDNC(5, 0.1)  #(1, 0.1)
    np.savetxt(path + 'SCPseDNC.txt', X)
    toc = time.clock()
    print('Coding time for SCPseDNC:%.3f minutes' % ((toc - tic) / 60))

    # 7 Series Correlation Pseudo Trinucleotide Composition
    tic = time.clock()
    X = GetSCPseTNC(10, 0.1)  #(6, 0.1)
    np.savetxt(path + 'SCPseTNC.txt', X)
    toc = time.clock()
    print('Coding time for SCPseTNC:%.3f minutes' % ((toc - tic) / 60))

    # 8 Dinucleotide-based auto covariance
    tic = time.clock()
    k = read_k('DNA', 'DAC', 0)
    # X = GetDAC(instances, k, 3, alphabet)
    X = GetDAC(instances, k, 8, alphabet)
    np.savetxt(path + 'DAC.txt', X)
    toc = time.clock()
    print('Coding time for DAC:%.3f minutes' % ((toc - tic) / 60))

    f0 = open((path + 'SpectrumProfile.txt'), 'w+')
    f1 = open((path + '1-SpectrumProfile.txt'), 'r')
    f2 = open((path + '2-SpectrumProfile.txt'), 'r')
    f3 = open((path + '3-SpectrumProfile.txt'), 'r')
    f4 = open((path + '4-SpectrumProfile.txt'), 'r')
    f5 = open((path + '5-SpectrumProfile.txt'), 'r')
    while True:
        F = []
        s1 = f1.readline()
        s1 = s1.strip('\n')
        F.extend(s1 + ' ')

        s2 = f2.readline()
        s2 = s2.strip('\n')
        F.extend(s2 + ' ')

        s3 = f3.readline()
        s3 = s3.strip('\n')
        F.extend(s3 + ' ')

        s4 = f4.readline()
        s4 = s4.strip('\n')
        F.extend(s4 + ' ')

        s5 = f5.readline()
        if not s5:
            break
        s5 = s5.strip('\n')
        F.extend(s5 + ' ')
        result = ''.join(F)
        f0.write(result + '\n')
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f0.close()

    f0 = open((path + 'MismatchProfile.txt'), 'w+')
    f1 = open((path + '(3, 1)-MismatchProfile.txt'), 'r')
    f2 = open((path + '(4, 1)-MismatchProfile.txt'), 'r')
    f3 = open((path + '(5, 1)-MismatchProfile.txt'), 'r')
    while True:
        F = []
        s1 = f1.readline()
        s1 = s1.strip('\n')
        F.extend(s1 + ' ')

        s2 = f2.readline()
        s2 = s2.strip('\n')
        F.extend(s2 + ' ')

        s3 = f3.readline()
        if not s3:
            break
        s3 = s3.strip('\n')
        F.extend(s3 + ' ')
        result = ''.join(F)
        f0.write(result + '\n')
    f1.close()
    f2.close()
    f3.close()
    f0.close()

    f0 = open((path + 'RevcKmer.txt'), 'w+')
    f1 = open((path + '1-RevcKmer.txt'), 'r')
    f2 = open((path + '2-RevcKmer.txt'), 'r')
    f3 = open((path + '3-RevcKmer.txt'), 'r')
    f4 = open((path + '4-RevcKmer.txt'), 'r')
    f5 = open((path + '5-RevcKmer.txt'), 'r')
    while True:
        F = []
        s1 = f1.readline()
        s1 = s1.strip('\n')
        F.extend(s1 + ' ')

        s2 = f2.readline()
        s2 = s2.strip('\n')
        F.extend(s2 + ' ')

        s3 = f3.readline()
        s3 = s3.strip('\n')
        F.extend(s3 + ' ')

        s4 = f4.readline()
        s4 = s4.strip('\n')
        F.extend(s4 + ' ')

        s5 = f5.readline()
        if not s5:
            break
        s5 = s5.strip('\n')
        F.extend(s5 + ' ')
        result = ''.join(F)
        f0.write(result + '\n')
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f0.close()

    f0 = open((path + 'Pse.txt'), 'w+')
    f1 = open((path + 'PCPseDNC.txt'), 'r')
    f2 = open((path + 'PCPseTNC.txt'), 'r')
    f3 = open((path + 'SCPseDNC.txt'), 'r')
    f4 = open((path + 'SCPseTNC.txt'), 'r')
    while True:
        F = []
        s1 = f1.readline()
        s1 = s1.strip('\n')
        F.extend(s1 + ' ')

        s2 = f2.readline()
        s2 = s2.strip('\n')
        F.extend(s2 + ' ')

        s3 = f3.readline()
        s3 = s3.strip('\n')
        F.extend(s3 + ' ')

        s4 = f4.readline()
        if not s4:
            break
        s4 = s4.strip('\n')
        F.extend(s4 + ' ')
        result = ''.join(F)
        f0.write(result + '\n')
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f0.close()


def ConstructPartitionOfSet(y, folds_num, seed):
    folds_temp = list(StratifiedKFold(y, n_folds=folds_num, shuffle=True, random_state=np.random.RandomState(seed)))
    folds = []
    c = range(folds_num)
    for i in range(folds_num):
        test_index = folds_temp[i][1]
        vali_index = folds_temp[(i + 1) % folds_num][1]
        train_index = array(list(set(folds_temp[i][0]) ^ set(vali_index)))
        folds.append((train_index, vali_index, test_index))
    return folds


def GetCrossValidation(X, y, feature, clf, folds):
    predicted_probas = -np.ones(len(y))
    predicted_labels = -np.ones(len(y))
    cv_round = 1
    for train_index, vali_index, test_index in folds:
        X_train, X_vali, X_test, y_train, y_vali = GetPartitionOfSamples_CV(X, y, train_index, vali_index,
                                                                         test_index)
        predict_test_proba, predict_test_label = MakePrediction(X_train, X_vali, X_test, y_train, y_vali, cv_round)
        predicted_probas[test_index] = predict_test_proba
        predicted_labels[test_index] = predict_test_label
        cv_round += 1
    auc_score, accuracy, sensitivity, specificity = EvaluatePerformances(y, predicted_probas, predicted_labels)
    return auc_score, accuracy, sensitivity, specificity


def GetPartitionOfSamples_CV(X, y, train_index, vali_index, test_index):
    y_train = y[train_index]
    y_vali = y[vali_index]
    X_train = X[train_index]
    X_vali = X[vali_index]
    X_test = X[test_index]
    return X_train, X_vali, X_test, y_train, y_vali


def MakePrediction(X_train, X_vali, X_test, y_train, y_vali, cv_round):
    classifier = clf.fit(X_train, y_train)
    joblib.dump(clf, path_train + feature + ".model")
    predict_vali_proba = classifier.predict_proba(X_vali)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_vali, predict_vali_proba, pos_label=1)
    auc_score = auc(fpr, tpr)
    print('Cross validation,round %d,the AUC is %.3f on validation dataset' % (cv_round, auc_score))
    predict_test_proba = classifier.predict_proba(X_test)[:, 1]
    predict_test_label = classifier.predict(X_test)
    return predict_test_proba, predict_test_label


def GetOptimalWeights(all_X, y, all_features, clf):
    proba_matrix = []
    # label_matrix=[]
    for i in range(len(all_features)):
        feature = all_features[i]
        X = all_X[i]
        print("..........................................................................")
        print("Cross validation,based on " + feature + ",beginning")
        tic = time.clock()
        proba_vector = -np.ones(len(y))
        k = 1
        for train_index, test_index in folds:
            print(".....fold %d....." % k)
            k += 1
            X_train, X_test, y_train = GetPartitionOfSamples(X, y, train_index, test_index)
            classifier = clf.fit(X_train, y_train)
            temp_test_proba = classifier.predict_proba(X_test)
            proba_vector[test_index] = temp_test_proba[:, 1]
        proba_matrix.append(proba_vector)
        toc = time.clock()
        print("Cross Validation,based on " + feature + ",running time:" + str((toc - tic) / 60.0) + " minutes")
        print('..........................................................................\n')
    proba_matrix = np.transpose(proba_matrix)
    tic = time.clock()
    optimal_weights = GeneticAlgorithm(proba_matrix, y)
    toc = time.clock()
    print('GA time:%.3f minutes' % ((toc - tic) / 60))
    np.savetxt(path_train + "OptimalWeights.txt", optimal_weights)


def GetPartitionOfSamples(X, y, train_index, test_index):
    y_train = y[train_index]
    X_train = X[train_index]
    X_test = X[test_index]
    return X_train, X_test, y_train


def GeneticAlgorithm(proba_matrix, y):
    global pops_num
    global generations
    global chr_length
    pops = GetPopulations(pops_num, chr_length)
    auc_scores = FitnessFunction(pops, proba_matrix, y)
    for k in range(generations):
        pops = UpdatePops(pops, auc_scores)
        auc_scores = FitnessFunction(pops, proba_matrix, y)
    max_auc = np.max(auc_scores)
    print('The maximum AUC is %.3f by using genetic algorithm' % max_auc)
    max_index = list(auc_scores).index(np.max(auc_scores))
    optimal_weights = pops[max_index]
    return optimal_weights


def GetPopulations(pops_num, chr_length):
    pops = []
    for i in range(pops_num - chr_length):
        temp_pop = [random.uniform(0, 1) for i in range(chr_length)]
        temp_pop = temp_pop / np.sum(temp_pop)
        pops.append(temp_pop)
    pops = array(pops)
    pops = np.vstack((np.eye(chr_length), pops))
    return pops


def FitnessFunction(pops, proba_matrix, y):
    auc_scores = []
    for i in range(np.shape(pops)[0]):
        weights = pops[i]
        combined_mean_proba = np.dot(proba_matrix, weights)
        fpr, tpr, thresholds = roc_curve(y, combined_mean_proba, pos_label=1)
        auc_scores.append(auc(fpr, tpr))
    auc_scores = array(auc_scores)
    return auc_scores


def UpdatePops(pops, auc_scores):
    global pops_num
    new_order = random.sample(range(pops_num), pops_num)
    for i in np.linspace(0, pops_num, num=pops_num / 2, endpoint=False, dtype=int):
        fmax = np.max(auc_scores)
        fmin = np.min(auc_scores)
        fmean = np.mean(auc_scores)

        select_index = new_order[i:i + 2]
        f = np.max(auc_scores[select_index])
        two_pops = pops[select_index].copy()

        probacrossover = (fmax - f) / (fmax - fmean) if f > fmean else 1
        cross_pops = Crossover(two_pops) if probacrossover > random.uniform(0, 1) else two_pops.copy()

        probamutation = 0.5 * (fmax - f) / (fmax - fmean) if f > fmean else (fmean - f) / (fmean - fmin)
        new_two_pops = Mutation(cross_pops) if probamutation > random.uniform(0, 1) else cross_pops.copy()

        pops[select_index] = new_two_pops.copy()
    return pops


def Crossover(two_pops):
    global chr_length
    cross_pops = two_pops.copy()
    crossposition = random.randint(2, chr_length - 3)
    cross_pops[0][0:crossposition] = two_pops[1][0:crossposition]
    cross_pops[1][0:crossposition] = two_pops[0][0:crossposition]
    cross_pops = Normalize(cross_pops)
    return cross_pops


def Mutation(cross_pops):
    global chr_length
    new_two_pops = cross_pops.copy()
    for i in range(2):
        mutation_num = random.randint(1, round(chr_length / 5))
        mutation_positions = random.sample(range(chr_length), mutation_num)
        new_two_pops[i][mutation_positions] = [random.uniform(0, 1) for j in range(mutation_num)]
    new_two_pops = Normalize(new_two_pops)
    return new_two_pops


def Normalize(two_pops):
    global chr_length
    for i in range(2):
        if np.sum(two_pops[i]) < 10 ** (-12):
            two_pops[i] = [random.uniform(0, 1) for j in range(chr_length)]
        two_pops[i] = two_pops[i] / np.sum(two_pops[i])
    return two_pops


def ModelPrediction(model, threshold, y_test):
    if model == "GAWE":
        proba_matrix = []
        label_matrix = []
        results = []
        for feature in all_features:
            print('Prediction, results for individual feature-based model: ' + feature)
            X_1 = np.loadtxt(path_test_Posis + feature + '.txt')
            X_2 = np.loadtxt(path_test_Negas + feature + '.txt')
            X = np.concatenate((X_1, X_2), axis=0)
            classifier = joblib.load(path_train + feature + ".model")
            temp_proba = classifier.predict_proba(X)[:, 1]
            temp_label = classifier.predict(X)
            auc_score, accuracy, sensitivity, specificity = EvaluatePerformances(y_test, temp_proba, temp_label)
            results.append([auc_score, accuracy, sensitivity, specificity])
            proba_matrix.append(temp_proba)
            label_matrix.append(temp_label)
        proba_matrix = np.transpose(proba_matrix)
        label_matrix = np.transpose(label_matrix)
        optimal_weights = np.loadtxt(path_train + "OptimalWeights.txt")
        predicted_proba = np.dot(proba_matrix, optimal_weights)
        # predicted_label = np.dot(label_matrix, optimal_weights) > threshold
        predicted_label = np.dot(proba_matrix, optimal_weights) > threshold
        SaveResults(results)
    else:
        X_1 = np.loadtxt(path_test_Posis + model + '.txt')
        X_2 = np.loadtxt(path_test_Negas + model + '.txt')
        X = np.concatenate((X_1, X_2), axis=0)
        classifier = joblib.load(path_train + model + ".model")
        predicted_proba = classifier.predict_proba(X)[:, 1]
        predicted_label = classifier.predict(X)
    return predicted_proba, predicted_label


def EvaluatePerformances(real_label, predicted_proba, predicted_label):
    fpr, tpr, thresholds = roc_curve(real_label, predicted_proba, pos_label=1)
    auc_score = auc(fpr, tpr)
    accuracy = accuracy_score(real_label, predicted_label)
    sensitivity = recall_score(real_label, predicted_label)
    specificity = (accuracy * len(real_label) - sensitivity * sum(real_label)) / (len(real_label) - sum(real_label))
    print('****AUC score:%.3f, accuracy:%.3f, sensitivity:%.3f, specificity:%.3f****\n' \
          % (auc_score, accuracy, sensitivity, specificity))

    return auc_score, accuracy, sensitivity, specificity


def SaveResults(results):
    results = array(results)
    df = DataFrame({'Feature': all_features, \
                    'AUC': results[:, 0], \
                    'ACC': results[:, 1], \
                    'SN': results[:, 2], \
                    'SP': results[:, 3]})
    df = df[['Feature', 'AUC', 'ACC', 'SN', 'SP']]
    # df.to_csv(path_train + 'IndividualFeatureResults('+'Predict'+train_plant+').csv',index=False)

###############################################################################

if __name__ == '__main__':

    # TAIR10:Posis:39518; Negas:39441
    # TIGR7:Posis:97232; Negas:165493
    global samples_file
    global alphabet
    global vidm
    global all_features
    global model
    global y_test
    global chr_length
    global pops_num
    global generations
    global user_indices_2
    global user_indices_3
    user_indices_2 = [[0.04, 0.06, 0.04, 0.05, 0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.04, 0.06, 0.03, 0.05, 0.04, 0.04],
                    [0.08, 0.07, 0.06, 0.10, 0.06, 0.06, 0.06, 0.06, 0.07, 0.07, 0.06, 0.07, 0.07, 0.07, 0.06, 0.08],
                    [0.07, 0.06, 0.06, 0.07, 0.05, 0.06, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.05, 0.06, 0.05, 0.07],
                    [6.69, 6.80, 3.47, 9.61, 2.00, 2.99, 2.71, 3.47, 4.27, 4.21, 2.99, 6.80, 1.85, 4.27, 2.00, 6.6],
                    [6.24, 2.91, 2.80, 4.66, 2.88, 2.67, 3.02, 2.80, 3.58, 2.66, 2.67, 2.91, 4.11, 3.58, 2.88, 6.24],
                    [21.34, 21.98, 17.48, 24.79, 14.51, 14.25, 14.66, 17.48, 18.41, 17.31, 14.25, 21.98, 14.24, 18.41,14.51, 21.34],
                    [1.05, 2.01, 3.60, 0.61, 5.60, 4.68, 6.02, 3.60, 2.44, 1.70, 4.68, 2.01, 3.50, 2.44, 5.60, 1.05],
                    [-1.26, 0.33, -1.66, 0.00, 0.14, -0.77, 0.00, -1.66, 1.44, 0.00, -0.77, 0.33, 0.00, 1.44, 0.14,-1.26],
                    [35.02, 31.53, 32.29, 30.72, 35.43, 33.54, 33.67, 32.29, 35.67, 34.07, 33.54, 31.53, 36.94, 35.67,35.43, 35.02],
                    [-0.18, -0.59, -0.22, -0.68, 0.48, -0.17, 0.44, -0.22, -0.05, -0.19, -0.17, -0.59, 0.04, -0.05,0.48, -0.18],
                    [0.01, -0.02, -0.02, 0.00, 0.01, 0.03, 0.00, -0.02, -0.01, 0.00, 0.03, -0.02, 0.00, -0.01, 0.01,0.01],
                    [3.25, 3.24, 3.32, 3.21, 3.37, 3.36, 3.29, 3.32, 3.30, 3.27, 3.36, 3.24, 3.39, 3.30, 3.37, 3.25],
                    [-1.00, -1.44, -1.28, -0.88, -1.45, -1.84, -2.17, -1.28, -1.30, -2.24, -1.84, -1.44, -0.58, -1.30,-1.45, -1.00],
                    [-7.60, -8.40, -7.80, -7.20, -8.50, -8.00, -10.60, -7.80, -8.20, -9.80, -8.00, -8.40, -7.20, -8.20,-8.50, -7.60],
                    [-21.30, -22.40, -21.00, -20.40, -22.70, -19.90, -27.20, -21.00, -22.20, -24.40, -19.90, -22.40,-21.30, -22.20, -22.70, -21.30]]
    user_indices_3 = [[-1.37, -0.70, -1.37, -0.70, -0.01, -0.01, -0.01, -0.01, -2.33, -0.14, -2.33, -0.14, 1.32, 1.32, 0.63, 1.32, -0.76, -0.34, -0.76, -0.34,
                     0.14, 0.14, 0.14, 0.14, -2.33, -2.33, -2.33, -2.33, 1.02, 1.02, 1.02, 1.02, -0.66, -0.81, -0.66, -0.81, 0.61, 0.61, 0.61, 0.61, 0.48,
                     0.48, 0.48, 0.48, 1.04, 1.04, 1.04, 1.04, 0.03, 0.27, 0.03, 0.27, -0.14, -0.14, -0.14, -0.14, 0.03, 0.3, 0.79, 0.3, 1.14, 1.14, 1.02, 1.02],
                    [1.78, 0.16, 1.78, 0.16, -0.19, -0.19, -0.19, -0.19, 1.78, 0.21, 1.78, 0.21, -1.00, -1.00, -0.71, -1.00, 0.16, -0.25, 0.16, -0.25, 0.04,
                     0.04, 0.04, 0.04, 1.78, 1.78, 1.78, 1.78, -1.00, -1.00, -1.00, -1.00, 1.78, 1.78, 1.78, 1.78, -0.25, -0.25, -0.25, -0.25, 0.04, 0.04,
                     0.04, 0.04, -0.83, -0.83, -0.83, -0.83, 0.04, -1.29, 0.04, -1.29, 0.21, 0.21, 0.21, 0.21, 0.04, -0.54, -1.93, -0.54, -1.41, -1.41, -1.00, -1.00],
                    [0.62, 0.15, 0.62, 0.15, -0.27, -0.27, -0.27, -0.27, 1.51, -0.71, 1.51, -0.71, 0.11, 0.11, 0.69, 0.11, 0.59, 0.91, 0.59, 0.91, -0.36, -0.36,
                     -0.36, -0.36, 1.51, 1.51, 1.51, 1.51, 0.11, 0.11, 0.11, 0.11, 0.62, 0.18, 0.62, 0.18, -1.22, -1.22, -1.22, -1.22, -1.67, -1.67, -1.67, -1.67,
                     -0.33, -0.33, -0.33, -0.33, -1.70, 1.70, -1.70, 1.70, -0.71, -0.71, -0.71, -0.71, -1.70, -0.20, 2.43, -0.20, 1.19, 1.19, 0.11, 0.11]]
    all_features = ['SpectrumProfile', 'MismatchProfile', 'RevcKmer', 'Pse', 'DAC']
    vdim = 35  # the fixed length of sequences for the PSSM feature
    folds_num = 10  # the number of folds for the cross validation
    seeds_num = 1  # the number of seeds for the partition of dataset
    chr_length = len(all_features)  # the length of chromosomes for the genetic algorithm
    pops_num = 1000  # the population size for the genetic algorithm
    generations = 500  # the generation for the genetic algorithm
    n_trees = 500  # the number of trees for the random forest
    threshold = 0.5
    plant = 'TIGR7'
    # plant = sys.argv[1]
    date = time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
    train_count = 8000
    test_count = 2000
    classifier_name = 'RF'
    alphabet = ['A', 'C', 'G', 'T']
    if plant == 'TAIR10':
        fp = 'TAIR10_DHSs.fas'
        fn = 'TAIR10_Non_DHSs.fas'
        posis_count = 39518
        negas_count = 39441
    elif plant == 'TIGR7':
        fp = 'TIGR7_DHSs.fas'
        fn = 'TIGR7_Non_DHSs.fas'
        posis_count = 97232
        negas_count = 165493

    y = array([1] * train_count + [0] * train_count)
    y_test = array([1] * test_count + [0] * test_count)

    path_train = 'D:\PyCharm\DHSs\\' + plant + '\\' + date + '\\' + str(train_count) + '\\'
    path_test = 'D:\PyCharm\DHSs\\' + plant + '\\' + date +  '\\' + str(test_count) + '\\'
    isExists = os.path.exists(path_train)
    if not isExists:
        f = os.makedirs(path_train)
    isExists = os.path.exists(path_test)
    if not isExists:
        f = os.makedirs(path_test)

    s = time.clock()

    start = time.clock()
    Posis_sample_train, Posis_sample_test = SeriesPartion(fp, posis_count, train_count, train_count+test_count, 'Posis')
    Negas_sample_train, Negas_sample_test = SeriesPartion(fn, negas_count, train_count, train_count+test_count, 'Negas')
    end = time.clock()
    print('Sampling time:%.3f minutes' % ((end - start) / 60))

    start = time.clock()
    samples_file = path_train + 'PosisAfterSampling.txt'
    path_1 = path_train + 'PosisFeatures\\'
    GetVariousClassFeatures(samples_file, path_1)
    samples_file = path_train + 'NegasAfterSampling.txt'
    path_2 = path_train + 'NegasFeatures\\'
    GetVariousClassFeatures(samples_file, path_2)
    samples_file = path_test + 'PosisAfterSampling.txt'
    path_3 = path_test + 'PosisFeatures\\'
    GetVariousClassFeatures(samples_file, path_3)
    samples_file = path_test + 'NegasAfterSampling.txt'
    path_4 = path_test + 'NegasFeatures\\'
    GetVariousClassFeatures(samples_file, path_4)
    end = time.clock()
    print('Getting Class Features time:%.3f minutes' % ((end - start) / 60))

    if classifier_name == 'RF':
        clf = RandomForestClassifier(random_state=1, n_estimators=n_trees)
    elif classifier_name == 'SVM':
        clf = svm.SVC(kernel='rbf', probability=True)
    elif classifier_name == 'LR':
        clf = LogisticRegression()
    average_results = 0
    for seed in range(1, 1 + seeds_num):
        print('################################# Seed %d ###################################' % seed)
        start = time.clock()
        folds = ConstructPartitionOfSet(y, folds_num, seed)
        all_X = []
        results = []
        for feature in all_features:
            print('.............................................................................')
            print('The Cross Validation based on feature:' + feature + ', beginning')
            tic = time.clock()
            X_1 = np.loadtxt(path_train + 'PosisFeatures\\' + feature + '.txt')
            X_2 = np.loadtxt(path_train + 'NegasFeatures\\' + feature + '.txt')
            X = np.concatenate((X_1, X_2), axis=0)
            all_X.append(X)
            print('The dimension of the ' + feature + ':%d' % len(X[0]))
            auc_score, accuracy, sensitivity, specificity = GetCrossValidation(X, y, feature, clf, folds)
            results.append([auc_score, accuracy, sensitivity, specificity])
            toc = time.clock()
            print('*****************************************************************************')
            print('The final results for feature:' + feature)
            print('****AUC score:%.3f, accuracy:%.3f, sensitivity:%.3f, specificity:%.3f****' \
                  % (auc_score, accuracy, sensitivity, specificity))
            print('Running time:%.3f mimutes' % ((toc - tic) / 60))
            print('*****************************************************************************')
            print('.............................................................................\n')
        results = array(results)
        df = DataFrame({'Feature': all_features, \
                        'AUC': results[:, 0], \
                        'ACC': results[:, 1], \
                        'SN': results[:, 2], \
                        'SP': results[:, 3]})
        df = df[['Feature', 'AUC', 'ACC', 'SN', 'SP']]
        df.to_csv(path_train + 'IndividualFeatureResults' + plant + 'CV(seed' + str(seed) + ')' + classifier_name + '.csv',
                  index=False)
        end = time.clock()
        print('Seed %d, total running time:%.3f minutes' % (seed, (end - start) / 60))
        print('#############################################################################')
        average_results += results
    average_results = average_results / seeds_num
    average_df = DataFrame({'Feature': all_features, \
                            'AUC': average_results[:, 0], \
                            'ACC': average_results[:, 1], \
                            'SN': average_results[:, 2], \
                            'SP': average_results[:, 3]})
    average_df = average_df[['Feature', 'AUC', 'ACC', 'SN', 'SP']]
    average_df.to_csv(path_train + 'IndividualFeatureAverageResults' + plant + 'CV(' + classifier_name + ').csv', index=False)

    start = time.clock()
    for seed in range(1, seeds_num + 1):
        folds = list(StratifiedKFold(y, n_folds=folds_num, shuffle=True, random_state=np.random.RandomState(seed)))
        GetOptimalWeights(all_X, y, all_features, clf)
    end = time.clock()
    print('Getting Optimal Weights time:%.3f minutes' % ((end - start) / 60))

    print('*****************************************************************************')
    start = time.clock()
    path_test_Posis = path_test + 'PosisFeatures\\'
    path_test_Negas = path_test + 'NegasFeatures\\'
    model = 'GAWE'
    predicted_proba, predicted_label = ModelPrediction(model, threshold, y_test)
    auc_score, accuracy, sensitivity, specificity = EvaluatePerformances(y_test, predicted_proba, predicted_label)
    df = DataFrame({'AUC': [auc_score], 'ACC': [accuracy], 'SN': [sensitivity], 'SP': [specificity]})
    df = df[['AUC', 'ACC', 'SN', 'SP']]
    df.to_csv(path_train + "Results(" + "Predict" + plant + "By" + model + ").csv", index=False)
    end = time.clock()
    print('Total prediction time:%.3f minutes' % ((end - start) / 60))
    print('*****************************************************************************\n')

    e = time.clock()
    print('Total running time:%.3f minutes' % ((e - s) / 60))