"""BIDMC / MIMIC-IV-ED table readers and feature derivations.

Functions for loading the core MIMIC-IV-ED tables and assembling the
harmonized encounter-level dataset (demographics, triage, visit/medication
history, outcomes) used by the FairED benchmark. This follows the MIMIC-IV-ED
preprocessing protocol of Xie et al., Sci Data 2022.
"""
import collections
from datetime import timedelta

import numpy as np
import pandas as pd


def convert_str_to_float(x):
    if isinstance(x, str):
        x_split = re.compile('[^a-zA-Z0-9-]').split(x.strip())
        if '-' in x_split[0]:
            x_split_dash = x_split[0].split('-')
            if len(x_split_dash) == 2 and x_split_dash[0].isnumeric() and x_split_dash[1].isnumeric():
                return (float(x_split_dash[0]) + float(x_split_dash[1])) / 2
            else:
                return np.nan
        else:
            if x_split[0].isnumeric():
                return float(x_split[0])
            else:
                return np.nan
    else:
        return x


def read_edstays_table(edstays_table_path):
    df_edstays = pd.read_csv(edstays_table_path)
    df_edstays['intime'] = pd.to_datetime(df_edstays['intime'])
    df_edstays['outtime'] = pd.to_datetime(df_edstays['outtime'])
    return df_edstays


def read_patients_table(patients_table_path):
    df_patients = pd.read_csv(patients_table_path)
    df_patients['dod'] = pd.to_datetime(df_patients['dod'])
    return df_patients


def read_admissions_table(admissions_table_path):
    df_admissions = pd.read_csv(admissions_table_path)
    df_admissions = df_admissions.rename(columns={"race": "ethnicity"})
    df_admissions =  df_admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime','ethnicity', 'edregtime','edouttime', 'insurance']]
    df_admissions['admittime'] = pd.to_datetime(df_admissions['admittime'])
    df_admissions['dischtime'] = pd.to_datetime(df_admissions['dischtime'])
    df_admissions['deathtime'] = pd.to_datetime(df_admissions['deathtime'])
    return df_admissions


def read_icustays_table(icustays_table_path):
    df_icu = pd.read_csv(icustays_table_path)
    df_icu['intime'] = pd.to_datetime(df_icu['intime'])
    df_icu['outtime'] = pd.to_datetime(df_icu['outtime'])
    return df_icu


def read_triage_table(triage_table_path):
    df_triage = pd.read_csv(triage_table_path)
    vital_rename_dict = {vital: '_'.join(['triage', vital]) for vital in ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity']}
    df_triage.rename(vital_rename_dict, axis=1, inplace=True)
    df_triage['triage_pain'] = df_triage['triage_pain'].apply(convert_str_to_float).astype(float)

    return df_triage


def read_diagnoses_table(diagnoses_table_path):
    df_diagnoses = pd.read_csv(diagnoses_table_path)
    return df_diagnoses


def read_vitalsign_table(vitalsign_table_path):
    df_vitalsign = pd.read_csv(vitalsign_table_path)
    vital_rename_dict = {vital: '_'.join(['ed', vital]) for vital in
                         ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'rhythm', 'pain']}
    df_vitalsign.rename(vital_rename_dict, axis=1, inplace=True)

    df_vitalsign['ed_pain'] = df_vitalsign['ed_pain'].apply(convert_str_to_float).astype(float)
    return df_vitalsign


def read_pyxis_table(pyxis_table_path):
    df_pyxis = pd.read_csv(pyxis_table_path)
    return df_pyxis


def merge_edstays_patients_on_subject(df_edstays,df_patients):
    if 'gender' in df_edstays.columns:
        df_edstays = pd.merge(df_edstays, df_patients[['subject_id', 'anchor_age', 'anchor_year','dod']], on = ['subject_id'], how='left')
    else:
        df_edstays = pd.merge(df_edstays, df_patients[['subject_id', 'anchor_age', 'gender', 'anchor_year','dod']], on = ['subject_id'], how='left')
    return df_edstays


def merge_edstays_admissions_on_subject(df_edstays ,df_admissions):
    df_edstays = pd.merge(df_edstays,df_admissions, on = ['subject_id', 'hadm_id'], how='left')
    return df_edstays


def merge_edstays_triage_on_subject(df_master ,df_triage):
    df_master = pd.merge(df_master,df_triage, on = ['subject_id', 'stay_id'], how='left')
    return df_master


def add_age(df_master):
    df_master['in_year'] = df_master['intime'].dt.year
    df_master['age'] = df_master['in_year'] - df_master['anchor_year'] + df_master['anchor_age']
    #df_master.drop(['anchor_age', 'anchor_year', 'in_year'],axis=1, inplace=True)
    return df_master


def add_inhospital_mortality(df_master):
    inhospital_mortality = df_master['dod'].notnull() & (df_master['dischtime'] >= df_master['dod'])
    df_master['outcome_inhospital_mortality'] = inhospital_mortality
    return df_master


def add_ed_los(df_master):
    ed_los = df_master['outtime'] - df_master['intime']
    df_master['ed_los'] = ed_los
    return df_master


def add_outcome_icu_transfer(df_master, df_icustays, timerange):
    timerange_delta = timedelta(hours = timerange)
    df_icustays_sorted = df_icustays[['subject_id', 'hadm_id', 'intime']].sort_values('intime')
    df_icustays_keep_first = df_icustays_sorted.groupby('hadm_id').first().reset_index()
    df_master_icu = pd.merge(df_master, df_icustays_keep_first, on = ['subject_id', 'hadm_id'], how='left', suffixes=('','_icu'))
    time_diff = (df_master_icu['intime_icu']- df_master_icu['outtime'])
    df_master_icu['time_to_icu_transfer'] = time_diff
    df_master_icu[''.join(['outcome_icu_transfer_', str(timerange), 'h'])] = time_diff <= timerange_delta
    # df_master_icu.drop(['intime_icu', 'time_to_icu_transfer'],axis=1, inplace=True)
    return df_master_icu


def fill_na_ethnicity(df_master): # requires df_master to be sorted 
    N = len(df_master)
    ethnicity_list= [float("NaN") for _ in range(N)]
    ethnicity_dict = {} # dict to store subejct ethnicity

    def get_filled_ethnicity(row):
        i = row.name
        if i % 10000 == 0:
            print('Process: %d/%d' % (i, N), end='\r')
        curr_eth = row['ethnicity']
        curr_subject = row['subject_id']
        prev_subject = df_master['subject_id'][i+1] if i< (N-1) else None

        if curr_subject not in ethnicity_dict.keys(): ## if subject ethnicity not stored yet, look ahead and behind 
            subject_ethnicity_list = []
            next_subject_idx = i+1
            prev_subject_idx = i-1
            next_subject= df_master['subject_id'][next_subject_idx] if next_subject_idx <= (N-1) else None
            prev_subject= df_master['subject_id'][prev_subject_idx] if prev_subject_idx >= 0 else None

            subject_ethnicity_list.append(df_master['ethnicity'][i]) ## add current ethnicity to list

            while prev_subject == curr_subject:
                subject_ethnicity_list.append(df_master['ethnicity'][prev_subject_idx])
                prev_subject_idx -= 1
                prev_subject= df_master['subject_id'][prev_subject_idx] if prev_subject_idx >= 0 else None

            while next_subject == curr_subject:
                subject_ethnicity_list.append(df_master['ethnicity'][next_subject_idx])
                next_subject_idx += 1
                next_subject= df_master['subject_id'][next_subject_idx] if next_subject_idx <= (N-1) else None
        
            eth_counter_list = collections.Counter(subject_ethnicity_list).most_common() #sorts counter and outputs list
            
            if len(eth_counter_list) == 0: ## no previous or next entries 
                subject_eth = curr_eth
            elif len(eth_counter_list) == 1: ## exactly one other ethnicity
                subject_eth = eth_counter_list.pop(0)[0] ## extract ethnicity from count tuple
            else:
                eth_counter_list = [x for x in eth_counter_list if pd.notna(x[0])] # remove any NA
                subject_eth = eth_counter_list.pop(0)[0]
            
            ethnicity_dict[curr_subject] = subject_eth ## store in dict
    
        if pd.isna(curr_eth): ## if curr_eth is na, fill with subject_eth from dict
            ethnicity_list[i]= ethnicity_dict[curr_subject]
        else:
            ethnicity_list[i]= curr_eth
            
    df_master.apply(get_filled_ethnicity, axis=1)
    print('Process: %d/%d' % (N, N), end='\r')
    df_master.loc[:,'ethnicity'] = ethnicity_list
    return df_master


def generate_past_ed_visits(df_master, timerange):
    #df_master = df_master.sort_values(['subject_id', 'intime']).reset_index()
    
    timerange_delta = timedelta(days=timerange)
    N = len(df_master)
    n_ed = [0 for _ in range(N)]

    def get_num_past_ed_visits(df):
        start = df.index[0]
        for i in df.index:
            if i % 10000 == 0:
                print('Process: %d/%d' % (i, N), end='\r')
            while df.loc[i, 'intime'] - df.loc[start, 'intime'] > timerange_delta:
                start += 1
            n_ed[i] = i - start

    grouped = df_master.groupby('subject_id')
    grouped.apply(get_num_past_ed_visits)
    print('Process: %d/%d' % (N, N), end='\r')

    df_master.loc[:, ''.join(['n_ed_', str(timerange), "d"])] = n_ed

    return df_master


def generate_past_admissions(df_master, df_admissions, timerange):
    df_admissions_sorted = df_admissions[df_admissions['subject_id'].isin(df_master['subject_id'].unique().tolist())][['subject_id', 'admittime']].copy()
    
    df_admissions_sorted.loc[:,'admittime'] = pd.to_datetime(df_admissions_sorted['admittime'])
    df_admissions_sorted.sort_values(['subject_id', 'admittime'], inplace=True)
    df_admissions_sorted.reset_index(drop=True, inplace=True)

    timerange_delta = timedelta(days=timerange)

    N = len(df_master)
    n_adm = [0 for _ in range(N)]

    def get_num_past_admissions(df):
        subject_id = df.iloc[0]['subject_id']
        if subject_id in grouped_adm.groups.keys():
            df_adm = grouped_adm.get_group(subject_id)
            start = end = df_adm.index[0]
            for i in df.index:
                if i % 10000 == 0:
                    print('Process: %d/%d' % (i, N), end='\r')
                while start < df_adm.index[-1] and df.loc[i, 'intime'] - df_adm.loc[start, 'admittime'] > timerange_delta:
                    start += 1
                end = start
                while end <= df_adm.index[-1] and \
                        (timerange_delta >= (df.loc[i, 'intime'] - df_adm.loc[end, 'admittime']) > timedelta(days=0)):
                    end += 1
                n_adm[i] = end - start

    grouped = df_master.groupby('subject_id')
    grouped_adm = df_admissions_sorted.groupby('subject_id')
    grouped.apply(get_num_past_admissions)
    print('Process: %d/%d' % (N, N), end='\r')

    df_master.loc[:,''.join(['n_hosp_', str(timerange), "d"])] = n_adm

    return df_master


def generate_past_icu_visits(df_master, df_icustays, timerange):
    df_icustays_sorted = df_icustays[df_icustays['subject_id'].isin(df_master['subject_id'].unique().tolist())][['subject_id', 'intime']].copy()
    df_icustays_sorted.sort_values(['subject_id', 'intime'], inplace=True)
    df_icustays_sorted.reset_index(drop=True, inplace=True)

    timerange_delta = timedelta(days=timerange)
    N = len(df_master)
    n_icu = [0 for _ in range(N)]
    def get_num_past_icu_visits(df):
        subject_id = df.iloc[0]['subject_id']
        if subject_id in grouped_icu.groups.keys():
            df_icu = grouped_icu.get_group(subject_id)
            start = end = df_icu.index[0]
            for i in df.index:
                if i % 10000 == 0:
                    print('Process: %d/%d' % (i, N), end='\r')
                while start < df_icu.index[-1] and df.loc[i, 'intime'] - df_icu.loc[start, 'intime'] > timerange_delta:
                    start += 1
                end = start
                while end <= df_icu.index[-1] and \
                        (timerange_delta >= (df.loc[i, 'intime'] - df_icu.loc[end, 'intime']) > timedelta(days=0)):
                    end += 1
                n_icu[i] = end - start

    grouped = df_master.groupby('subject_id')
    grouped_icu = df_icustays_sorted.groupby('subject_id')
    grouped.apply(get_num_past_icu_visits)
    print('Process: %d/%d' % (N, N), end='\r')

    df_master.loc[:,''.join(['n_icu_', str(timerange), "d"])] = n_icu

    return df_master


def generate_future_ed_visits(df_master, next_ed_visit_timerange):
    N = len(df_master)
    time_of_next_ed_visit = [float("NaN") for _ in range(N)]
    time_to_next_ed_visit = [float("NaN") for _ in range(N)]
    outcome_ed_revisit = [False for _ in range(N)]

    timerange_delta = timedelta(days = next_ed_visit_timerange)

    curr_subject=None
    next_subject=None

    def get_future_ed_visits(row):
        i = row.name
        if i % 10000 == 0:
            print('Process: %d/%d' % (i, N), end='\r')
        curr_subject = row['subject_id']
        next_subject= df_master['subject_id'][i+1] if i< (N-1) else None

        if curr_subject == next_subject:
            curr_outtime = row['outtime']
            next_intime = df_master['intime'][i+1]
            next_intime_diff = next_intime - curr_outtime

            time_of_next_ed_visit[i] = next_intime
            time_to_next_ed_visit[i] = next_intime_diff
            outcome_ed_revisit[i] = next_intime_diff < timerange_delta

    df_master.apply(get_future_ed_visits, axis=1)
    print('Process: %d/%d' % (N, N), end='\r')

    df_master.loc[:,'next_ed_visit_time'] = time_of_next_ed_visit
    df_master.loc[:,'next_ed_visit_time_diff'] = time_to_next_ed_visit
    df_master.loc[:,''.join(['outcome_ed_revisit_', str(next_ed_visit_timerange), "d"])] = outcome_ed_revisit

    return df_master


def generate_numeric_timedelta(df_master):
    N = len(df_master)
    ed_los_hours = [float("NaN") for _ in range(N)]
    time_to_icu_transfer_hours = [float("NaN") for _ in range(N)]
    next_ed_visit_time_diff_days = [float("NaN") for _ in range(N)]
    
    def get_numeric_timedelta(row):
        i = row.name
        if i % 10000 == 0:
            print('Process: %d/%d' % (i, N), end='\r')
        curr_subject = row['subject_id']
        curr_ed_los = row['ed_los']
        curr_time_to_icu_transfer = row['time_to_icu_transfer']
        curr_next_ed_visit_time_diff = row['next_ed_visit_time_diff']
        

        ed_los_hours[i] = round(curr_ed_los.total_seconds() / (60*60),2) if not pd.isna(curr_ed_los) else curr_ed_los
        time_to_icu_transfer_hours[i] = round(curr_time_to_icu_transfer.total_seconds() / (60*60),2) if not pd.isna(curr_time_to_icu_transfer) else curr_time_to_icu_transfer
        next_ed_visit_time_diff_days[i] = round(curr_next_ed_visit_time_diff.total_seconds() / (24*60*60), 2) if not pd.isna(curr_next_ed_visit_time_diff) else curr_next_ed_visit_time_diff
    

    df_master.apply(get_numeric_timedelta, axis=1)
    print('Process: %d/%d' % (N, N), end='\r')
    
    df_master.loc[:,'ed_los_hours'] = ed_los_hours
    df_master.loc[:,'time_to_icu_transfer_hours'] = time_to_icu_transfer_hours
    df_master.loc[:,'next_ed_visit_time_diff_days'] = next_ed_visit_time_diff_days

    return df_master


def merge_vitalsign_info_on_edstay(df_master, df_vitalsign, options=[]):
    df_vitalsign.sort_values('charttime', inplace=True)

    grouped = df_vitalsign.groupby(['stay_id'])

    for option in options:
        method = getattr(grouped, option, None)
        assert method is not None, "Invalid option. " \
                                   "Should be a list of values from 'max', 'min', 'median', 'mean', 'first', 'last'. " \
                                   "e.g. ['median', 'last']"
        df_vitalsign_option = method(numeric_only=True)
        df_vitalsign_option.rename({name: '_'.join([name, option]) for name in
                                    ['ed_temperature', 'ed_heartrate', 'ed_resprate', 'ed_o2sat', 'ed_sbp', 'ed_dbp', 'ed_pain']},
                                   axis=1,
                                   inplace=True)
        df_master = pd.merge(df_master, df_vitalsign_option, on=['subject_id', 'stay_id'], how='left')

    return df_master


def merge_med_count_on_edstay(df_master, df_pyxis):
    df_pyxis_fillna = df_pyxis.copy()
    df_pyxis_fillna['gsn'].fillna(df_pyxis['name'], inplace=True)
    grouped = df_pyxis_fillna.groupby(['stay_id'])
    df_medcount = grouped['gsn'].nunique().reset_index().rename({'gsn': 'n_med'}, axis=1)
    df_master = pd.merge(df_master, df_medcount, on='stay_id', how='left')
    df_master.fillna({'n_med': 0}, inplace=True)
    return df_master


def merge_medrecon_count_on_edstay(df_master, df_medrecon):
    df_medrecon_fillna = df_medrecon.copy()
    df_medrecon_fillna['gsn'].fillna(df_medrecon['name'])
    grouped = df_medrecon_fillna.groupby(['stay_id'])
    df_medcount = grouped['gsn'].nunique().reset_index().rename({'gsn': 'n_medrecon'}, axis=1)
    df_master = pd.merge(df_master, df_medcount, on='stay_id', how='left')
    df_master.fillna({'n_medrecon': 0}, inplace=True)
    return df_master
