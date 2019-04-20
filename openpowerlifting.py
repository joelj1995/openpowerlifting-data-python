import pandas
import numpy as np
import datetime

import math
from collections import defaultdict
from collections import Counter


IPF_WC_WOMEN_NUM = [47, 52, 57, 63, 72, 84]
IPF_WC_MEN_NUM = [59, 66, 74, 83, 93, 105, 120]

IPF_WC_WOMEN = [str(x) for x in IPF_WC_WOMEN_NUM] + ['84+']
IPF_WC_MEN = [str(x) for x in IPF_WC_MEN_NUM] + ['120+']


def load(file_name='openpowerlifting.csv'):
    return pandas.read_csv(file_name, dtype={
    'Name': str, 
    'Sex': str}, parse_dates=[34], index_col=0)


def group_by_name(df, column):
    # Returns a dictionary indexed by the lifter name
    # to a list of their competition records
    d = defaultdict(list)
    for index, item in df[column].iteritems():
        d[index].append(item)
    return d


def get_bests(d):
    # Accepts dictionary d, generated with group_by_name
    # Returns a list containing the max in each lifter's list
    bests = []
    for key in d:
        best = max(d[key])
        if best > 0:
            bests.append(best)
    return bests


def cohen_effect_size(group1, group2):
    diff = np.average(group1) - np.average(group2)

    var1 = np.var(group1)
    var2 = np.var(group2)
    n1, n2 = len(group1), len(group2)

    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / math.sqrt(pooled_var)
    return d


def cov(xs, ys, meanx=None, meany=None):
    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)

    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov


def corr(xs, ys):
    meanx = np.mean(xs)
    meany = np.mean(ys)

    varx = np.var(xs)
    vary = np.var(ys)

    corr = cov(xs, ys, meanx, meany) / math.sqrt(varx * vary)
    return corr


def QueryFullPower(df):
    return df[df.Event == 'SBD']

def QueryRaw(df):
    return df[df.Equipment == 'Raw']

def QuerySinglePly(df):
    return df[df.Equipment == 'Single-ply']

def QueryOpen(df):
    return df[df.AgeClass == '24-34']

def QueryMale(df):
    return df[df.Sex == 'M']

def QueryFemale(df):
    return df[df.Sex == 'F']

def QueryFederation(df, fed):
    return df[df.Federation == fed]

def QueryWeightClass(df, wc):
    return df[df.WeightClassKg == wc]

def QueryCPU105FullPower(df):
    df = QueryFullPower(df)
    df = QueryRaw(df)
    df = QueryFederation(df, 'CPU')
    df = QueryWeightClass(df, '105')
    return df

def QueryRawIPFMenFullPower(df):
    df = QueryFullPower(df)
    df = QueryRaw(df)
    df = QueryMale(df)
    df = QueryFederation(df, 'IPF')
    return df

def QuerySingleIPFMenFullPower(df):
    df = QueryFullPower(df)
    df = QuerySinglePly(df)
    df = QueryMale(df)
    df = QueryFederation(df, 'IPF')
    return df