import re
from MSK_featurename_list_060726 import *
from itertools import combinations, permutations, product
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import math

def patient_specific_side_effect_exclusion_event_adjudication(DMP_ID, all_test_dates, all_dates, data_dict_lab_dict, MED_DMP_ID_TO_START_END_DATE_DICT, DMP_IDs_to_lastcontactdate_dict, DMP_ID_TO_INCLUDED_MED_DATES, task='anemia_risk', all_vital_dates=[], include_gcsf=False, anc=False):
    """adjudicating whether a patient should be removed, and if not this patient is a positive/negative case for the task + acquire data for the task;
        return three elements: removal_dict, data_input_dates, cutoffdate, y"""
    removal_dict = {
        # "no_pre_med_start_labs_removal": False,
    "lack_2_rel_past_removal": False, "lack_1_rel_future_removal": False, "baseline_side_effect_removal": False, "censor_removal": False}
    if not bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', task)):
        raise Exception("Not an included side effect prediction task")
    else:
        DMP_ID_to_DATES_to_TESTSVALUES_DICT = data_dict_lab_dict['lab_def_dict']
        post_med_start_dates = [date for date in all_test_dates if date > MED_DMP_ID_TO_START_END_DATE_DICT[DMP_ID][0] and date <= MED_DMP_ID_TO_START_END_DATE_DICT[DMP_ID][1]]
        pre_med_start_dates = [date for date in all_test_dates if date <= MED_DMP_ID_TO_START_END_DATE_DICT[DMP_ID][0]] # this is used for fetching actual feature values, can be as far back as needed
        if bool(re.match(r'^(febrile_neutropenia)(_next_risk|_risk)?$', task)):
            DMP_ID_to_DATES_to_VITALS_DICT = data_dict_lab_dict['vitals_def_dict']
            post_med_start_vital_dates = [date for date in all_vital_dates if date > MED_DMP_ID_TO_START_END_DATE_DICT[DMP_ID][0] and date <= MED_DMP_ID_TO_START_END_DATE_DICT[DMP_ID][1]]
            pre_med_start_vital_dates = [date for date in all_vital_dates if date <= MED_DMP_ID_TO_START_END_DATE_DICT[DMP_ID][0]] # this is used for fetching actual feature values, can be as far back as needed
            pre_med_start_vital_dates = [date for date in pre_med_start_vital_dates if date >= MED_DMP_ID_TO_START_END_DATE_DICT[DMP_ID][0]-180]


        ## ensure past_dates, used to compute baseline values, are at most 180 days prior to the start date of the medication
        baseline_judging_dates = [date for date in pre_med_start_dates if date >= MED_DMP_ID_TO_START_END_DATE_DICT[DMP_ID][0]-180]

        if bool(re.match(r'^(anemia)(_next_risk|_risk)?$', task)):
            return anemia_adjudication(DMP_ID, removal_dict, pre_med_start_dates, post_med_start_dates, baseline_judging_dates, DMP_ID_to_DATES_to_TESTSVALUES_DICT, MED_DMP_ID_TO_START_END_DATE_DICT, DMP_IDs_to_lastcontactdate_dict, all_dates, DMP_ID_TO_INCLUDED_MED_DATES, task)
        elif bool(re.match(r'^(AKI)(_next_risk|_risk)?$', task)):
            return AKI_adjudication(DMP_ID, removal_dict, pre_med_start_dates, post_med_start_dates, baseline_judging_dates, DMP_ID_to_DATES_to_TESTSVALUES_DICT, MED_DMP_ID_TO_START_END_DATE_DICT, DMP_IDs_to_lastcontactdate_dict, all_dates, DMP_ID_TO_INCLUDED_MED_DATES, task)
        elif bool(re.match(r'^(neutropenia)(_next_risk|_risk)?$', task)):
            return neutropenia_FN_adjudication(DMP_ID, removal_dict, pre_med_start_dates, post_med_start_dates, baseline_judging_dates, DMP_ID_to_DATES_to_TESTSVALUES_DICT, MED_DMP_ID_TO_START_END_DATE_DICT, DMP_IDs_to_lastcontactdate_dict, all_dates, DMP_ID_TO_INCLUDED_MED_DATES, task, include_gcsf=include_gcsf, anc=anc)
        elif bool(re.match(r'^(febrile_neutropenia)(_next_risk|_risk)?$', task)):
            return neutropenia_FN_adjudication(DMP_ID, removal_dict, pre_med_start_dates, post_med_start_dates, baseline_judging_dates, DMP_ID_to_DATES_to_TESTSVALUES_DICT, MED_DMP_ID_TO_START_END_DATE_DICT, DMP_IDs_to_lastcontactdate_dict, all_dates, DMP_ID_TO_INCLUDED_MED_DATES, task, post_med_start_vital_dates, pre_med_start_vital_dates, DMP_ID_to_DATES_to_VITALS_DICT, include_gcsf=include_gcsf, anc=anc)
        elif bool(re.match(r'^(Liver_toxicity)(_next_risk|_risk)?$', task)):
            return LT_adjudication(DMP_ID, removal_dict, pre_med_start_dates, post_med_start_dates, baseline_judging_dates, DMP_ID_to_DATES_to_TESTSVALUES_DICT, MED_DMP_ID_TO_START_END_DATE_DICT, DMP_IDs_to_lastcontactdate_dict, all_dates, DMP_ID_TO_INCLUDED_MED_DATES, task)


def anemia_adjudication(DMP_ID, removal_dict, pre_med_start_dates, post_med_start_dates, baseline_judging_dates, DMP_ID_to_DATES_to_TESTSVALUES_DICT, MED_DMP_ID_TO_START_END_DATE_DICT, DMP_IDs_to_lastcontactdate_dict, all_dates, DMP_ID_TO_INCLUDED_MED_DATES, task):
    post_med_start_date_to_HGB_levels = {date: DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID][date][ANEMIA_INDEX] for date in post_med_start_dates if date in DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID]}
    post_med_start_date_to_HGB_levels = {date: value for date, value in post_med_start_date_to_HGB_levels.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
    post_med_start_HGB_levels = list(post_med_start_date_to_HGB_levels.values())
    baseline_date_to_HGB_levels = {date: DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID][date][ANEMIA_INDEX] for date in baseline_judging_dates if date in DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID]}
    baseline_date_to_HGB_levels = {date: value for date, value in baseline_date_to_HGB_levels.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
    baseline_HGB_levels = list(baseline_date_to_HGB_levels.values())
    
    ## 2. check for lack of 2 relevant measurements prior to med start
    if len(baseline_HGB_levels) < 2:
        removal_dict["lack_2_rel_past_removal"] = True
        return {0: (removal_dict, [], '', '', '', '')}
    ## 3. check for baseline side effect
    if min(baseline_HGB_levels) < 8:
        removal_dict["baseline_side_effect_removal"] = True
        return {0: (removal_dict, [], '', '', '', '')}
    else:
        # TODO: for each medication infusion (specified in DMP_ID_TO_INCLUDED_MED_DATES), check whether a patient has anemia within the minimum of 30D (washout) and before the next infusion
        ### note 1: each patient is counted multiple times, so this should output a dict where the key is the infusion # and the values are dicts with removal_dict, input_data_dates, input_end_date, y
        ### note 2: a patient starts to get censored if their last contact date is within 30 days of an infusion and there is no positive anemia event before the last contact date (and after the infusion)
        if len(post_med_start_HGB_levels) == 0:
            removal_dict["lack_1_rel_future_removal"] = True
            return {0: (removal_dict, [], '', '', '', '')}

        # PASS 1: Find when anemia occurs (which idx has y==1)
        ## also get the hgb measurements within all post med infusion windows (up to next infusion or end of washout of each infusion)
        whole_med_period_hgb_measurements = {}
        anemia_idx = None
        temp_y_list = []

        for idx, med_start_date in enumerate(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]):
            inf_window_measurements = {date: value for date, value in post_med_start_date_to_HGB_levels.items() if date >= med_start_date and (idx == len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 or date < min(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][idx+1], DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][idx]+30))}
            whole_med_period_hgb_measurements.update(inf_window_measurements)

            current_hgb_measurements = {date: value for date, value in post_med_start_date_to_HGB_levels.items() if date >= med_start_date and (idx == len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 or date < min(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][idx+1], DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][idx]+30))}

            if len(current_hgb_measurements) > 0 and min(current_hgb_measurements.values()) < 8:
                y = 1
                anemia_idx = idx  # Record where anemia happens
            else:
                y = 0

            temp_y_list.append((y, current_hgb_measurements))

            if y == 1:
                break  # Early termination when anemia found

        # Determine entire_y
        if len(whole_med_period_hgb_measurements) > 0 and min(whole_med_period_hgb_measurements.values()) < 8:
            entire_y = 1
        else:
            entire_y = 0

        # PASS 2: Build results with correct time_to_event
        results = {}
        for idx, med_start_date in enumerate(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]):
            # 1. check if the last contact date is outside of the washout period, if so censor and break out the for loop
            if idx == len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 and DMP_IDs_to_lastcontactdate_dict[DMP_ID] < med_start_date + 30:
                removal_dict_instance = removal_dict.copy()
                removal_dict_instance["censor_removal"] = True
                results[idx] = (removal_dict_instance, [], '', '', '', '')
                return results

            # Retrieve y from pass 1
            if idx >= len(temp_y_list):
                break
            y, current_hgb_measurements = temp_y_list[idx]

            # Calculate correct time_to_event (inclusive of current infusion)
            if anemia_idx is not None:
                # Anemia occurs - count cycles from current idx to anemia_idx (inclusive)
                current_num_durations_to_anemia_or_cutoff = (anemia_idx - idx) + 1
            else:
                # No anemia - count remaining cycles (censored, inclusive)
                current_num_durations_to_anemia_or_cutoff = (len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]) - idx) + 1

            input_data_dates = [date for date in all_dates if date <= med_start_date]
            results[idx] = (removal_dict.copy(), input_data_dates, med_start_date, y, entire_y, current_num_durations_to_anemia_or_cutoff)
            if y == 1:
                return results
        return results

def LT_adjudication(DMP_ID, removal_dict, pre_med_start_dates, post_med_start_dates, baseline_judging_dates, DMP_ID_to_DATES_to_TESTSVALUES_DICT, MED_DMP_ID_TO_START_END_DATE_DICT, DMP_IDs_to_lastcontactdate_dict, all_dates, DMP_ID_TO_INCLUDED_MED_DATES, task):
    ### Bilirubin for sinusoidal obstruction syndrome and rest for test-specific elevations that represent toxicity
    post_med_start_date_to_Bilirubin_dict = {date: DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID][date][BILIRUBIN_INDEX] for date in post_med_start_dates if date in DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID]}
    post_med_start_date_to_AST_dict = {date: DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID][date][AST_INDEX] for date in post_med_start_dates if date in DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID]}
    post_med_start_date_to_ALT_dict = {date: DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID][date][ALT_INDEX] for date in post_med_start_dates if date in DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID]}
    post_med_start_date_to_ALP_dict = {date: DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID][date][ALP_INDEX] for date in post_med_start_dates if date in DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID]}
    baseline_date_to_Bilirubin_dict = {date: DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID][date][BILIRUBIN_INDEX] for date in baseline_judging_dates if date in DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID]}
    baseline_date_to_AST_dict = {date: DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID][date][AST_INDEX] for date in baseline_judging_dates if date in DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID]}
    baseline_date_to_ALT_dict = {date: DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID][date][ALT_INDEX] for date in baseline_judging_dates if date in DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID]}
    baseline_date_to_ALP_dict = {date: DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID][date][ALP_INDEX] for date in baseline_judging_dates if date in DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID]}
    
    # remove NA/NANs
    post_med_start_date_to_Bilirubin_dict = {date: value for date, value in post_med_start_date_to_Bilirubin_dict.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
    post_med_start_date_to_AST_dict = {date: value for date, value in post_med_start_date_to_AST_dict.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
    post_med_start_date_to_ALT_dict = {date: value for date, value in post_med_start_date_to_ALT_dict.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
    post_med_start_date_to_ALP_dict = {date: value for date, value in post_med_start_date_to_ALP_dict.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
    baseline_date_to_Bilirubin_dict = {date: value for date, value in baseline_date_to_Bilirubin_dict.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
    baseline_date_to_AST_dict = {date: value for date, value in baseline_date_to_AST_dict.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
    baseline_date_to_ALT_dict = {date: value for date, value in baseline_date_to_ALT_dict.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
    baseline_date_to_ALP_dict = {date: value for date, value in baseline_date_to_ALP_dict.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
    
    ## 2. check for lack of 2 relevant measurements prior to med start
    if len(baseline_date_to_Bilirubin_dict) < 2 or len(baseline_date_to_AST_dict) < 2 or len(baseline_date_to_ALT_dict) < 2 or len(baseline_date_to_ALP_dict) < 2:
        removal_dict["lack_2_rel_past_removal"] = True
        return {0: (removal_dict, [], '', '', '', '')}
        
    ## 3. check for baseline side effect
    baseline_sos_status = min(baseline_date_to_Bilirubin_dict.values()) > 5
    # Sort by date to get chronological first value
    sorted_AST_values = [v for k, v in sorted(baseline_date_to_AST_dict.items())]
    sorted_ALP_values = [v for k, v in sorted(baseline_date_to_ALP_dict.items())]
    sorted_ALT_values = [v for k, v in sorted(baseline_date_to_ALT_dict.items())]
    AST_baseline, ALP_baseline, ALT_baseline = sorted_AST_values[0], sorted_ALP_values[0], sorted_ALT_values[0]
    ULN_AST, ULN_ALP, ULN_ALT = 33, 147, 36
    AST_baseline, ALP_baseline, ALT_baseline = max(AST_baseline, ULN_AST), max(ALP_baseline, ULN_ALP), max(ALT_baseline, ULN_ALT)
    baseline_nonfirst_AST_values, baseline_nonfirst_ALP_values, baseline_nonfirst_ALT_values = sorted_AST_values[1:], sorted_ALP_values[1:], sorted_ALT_values[1:]
    baseline_othertest_status = any(val > 5*AST_baseline for val in baseline_nonfirst_AST_values) or \
                                any(val > 5*ALP_baseline for val in baseline_nonfirst_ALP_values) or \
                                any(val > 5*ALT_baseline for val in baseline_nonfirst_ALT_values)
    if baseline_sos_status or baseline_othertest_status:
        # print("DMP_ID with baseline LT: ", DMP_ID, baseline_sos_status, baseline_othertest_status, baseline_date_to_Bilirubin_dict, baseline_date_to_AST_dict, baseline_date_to_ALT_dict, baseline_date_to_ALP_dict)
        removal_dict["baseline_side_effect_removal"] = True
        return {0: (removal_dict, [], '', '', '', '')}
    else:
        # Check for lack of future measurements
        if len(post_med_start_date_to_Bilirubin_dict) == 0 or len(post_med_start_date_to_AST_dict) == 0 or len(post_med_start_date_to_ALT_dict) == 0 or len(post_med_start_date_to_ALP_dict) == 0:
            removal_dict["lack_1_rel_future_removal"] = True
            return {0: (removal_dict, [], '', '', '', '')}
            
        # recalculate baseline using all past values
        AST_baseline, ALP_baseline, ALT_baseline = sum(list(baseline_date_to_AST_dict.values()))/len(baseline_date_to_AST_dict), sum(list(baseline_date_to_ALP_dict.values()))/len(baseline_date_to_ALP_dict), sum(list(baseline_date_to_ALT_dict.values()))/len(baseline_date_to_ALT_dict)
        AST_baseline, ALP_baseline, ALT_baseline = max(AST_baseline, ULN_AST), max(ALP_baseline, ULN_ALP), max(ALT_baseline, ULN_ALT)              
        
        ## whole med period measurements (for entire_y)
        whole_med_period_bilirubin_measurements = {}
        whole_med_period_ast_measurements = {}
        whole_med_period_alt_measurements = {}
        whole_med_period_alp_measurements = {}
        for inf_idx, inf_date in enumerate(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]):
            inf_window_measurements_bilirubin = {date: value for date, value in post_med_start_date_to_Bilirubin_dict.items() if date >= inf_date and (inf_idx == len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 or date < min(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx+1], DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx]+90))}
            inf_window_measurements_ast = {date: value for date, value in post_med_start_date_to_AST_dict.items() if date >= inf_date and (inf_idx == len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 or date < min(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx+1], DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx]+90))}
            inf_window_measurements_alt = {date: value for date, value in post_med_start_date_to_ALT_dict.items() if date >= inf_date and (inf_idx == len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 or date < min(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx+1], DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx]+90))}
            inf_window_measurements_alp = {date: value for date, value in post_med_start_date_to_ALP_dict.items() if date >= inf_date and (inf_idx == len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 or date < min(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx+1], DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx]+90))}
            whole_med_period_bilirubin_measurements.update(inf_window_measurements_bilirubin)
            whole_med_period_ast_measurements.update(inf_window_measurements_ast)
            whole_med_period_alt_measurements.update(inf_window_measurements_alt)
            whole_med_period_alp_measurements.update(inf_window_measurements_alp)

        ## entire med period status
        entire_sos_status = len(whole_med_period_bilirubin_measurements) > 0 and min(whole_med_period_bilirubin_measurements.values()) > 5
        entire_othertest_status = (len(whole_med_period_ast_measurements) > 0 and any(val > 5*AST_baseline for val in whole_med_period_ast_measurements.values())) or \
                                    (len(whole_med_period_alt_measurements) > 0 and any(val > 5*ALT_baseline for val in whole_med_period_alt_measurements.values())) or \
                                    (len(whole_med_period_alp_measurements) > 0 and any(val > 5*ALP_baseline for val in whole_med_period_alp_measurements.values()))
        if entire_sos_status or entire_othertest_status:
            entire_y = 1
        else:
            entire_y = 0

        # PASS 1: Find when LT occurs (which idx has y==1)
        lt_idx = None
        temp_y_list = []

        for idx, med_start_date in enumerate(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]):
            next_infusion_date = DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][idx+1] if idx < len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 else float('inf')
            end_window = min(next_infusion_date, med_start_date + 90)

            current_bilirubin_measurements = {date: value for date, value in post_med_start_date_to_Bilirubin_dict.items() if date >= med_start_date and date < end_window}
            current_ast_measurements = {date: value for date, value in post_med_start_date_to_AST_dict.items() if date >= med_start_date and date < end_window}
            current_alt_measurements = {date: value for date, value in post_med_start_date_to_ALT_dict.items() if date >= med_start_date and date < end_window}
            current_alp_measurements = {date: value for date, value in post_med_start_date_to_ALP_dict.items() if date >= med_start_date and date < end_window}

            current_sos_status = len(current_bilirubin_measurements) > 0 and min(current_bilirubin_measurements.values()) > 5
            current_othertest_status = (len(current_ast_measurements) > 0 and any(val > 5*AST_baseline for val in current_ast_measurements.values())) or \
                                      (len(current_alt_measurements) > 0 and any(val > 5*ALT_baseline for val in current_alt_measurements.values())) or \
                                      (len(current_alp_measurements) > 0 and any(val > 5*ALP_baseline for val in current_alp_measurements.values()))

            if current_sos_status or current_othertest_status:
                y = 1
                lt_idx = idx  # Record where LT happens
            else:
                y = 0

            temp_y_list.append(y)

            if y == 1:
                break  # Early termination when LT found

        # PASS 2: Build results with correct time_to_event
        results = {}
        for idx, med_start_date in enumerate(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]):
            # 1. check if the last contact date is outside of the washout period, if so censor and break out the for loop
            if idx == len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 and DMP_IDs_to_lastcontactdate_dict[DMP_ID] < med_start_date + 90:
                removal_dict_instance = removal_dict.copy()
                removal_dict_instance["censor_removal"] = True
                results[idx] = (removal_dict_instance, [], '', '', entire_y, '')
                return results

            # Retrieve y from pass 1
            if idx >= len(temp_y_list):
                break
            y = temp_y_list[idx]

            # Calculate correct time_to_event (inclusive of current infusion)
            if lt_idx is not None:
                # LT occurs - count cycles from current idx to lt_idx (inclusive)
                current_num_durations_to_LT_or_cutoff = (lt_idx - idx) + 1
            else:
                # No LT - count remaining cycles (censored, inclusive)
                current_num_durations_to_LT_or_cutoff = (len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]) - idx) + 1

            input_data_dates = [date for date in all_dates if date <= med_start_date]
            results[idx] = (removal_dict.copy(), input_data_dates, med_start_date, y, entire_y, current_num_durations_to_LT_or_cutoff)

            if y == 1:
                return results

        return results

def match_neutrophil_and_temp(neut_dict, temp_dict, days=2):
    result = {}
    for n_date, n_val in neut_dict.items():
        for t_date, t_val in temp_dict.items():
            if abs(n_date - t_date) <= days:
                result.setdefault(min(n_date, t_date), []).append((n_val, t_val))
    return result

def define_FN(FN_dict, anc_cutoff=1000.0, fever_c=38.0):
    """FN_dict: {date: [(anc, temp_c), ...], ...}"""
    hit_dates = []
    for date, pairs in (FN_dict or {}).items():
        for anc, temp in (pairs or []):
            if anc < anc_cutoff and temp >= fever_c:
                hit_dates.append(date)
                break
    return (True, min(hit_dates)) if hit_dates else (False, None)

def neutropenia_FN_adjudication(DMP_ID, removal_dict, pre_med_start_dates, post_med_start_dates, baseline_judging_dates, DMP_ID_to_DATES_to_TESTSVALUES_DICT, MED_DMP_ID_TO_START_END_DATE_DICT, DMP_IDs_to_lastcontactdate_dict, all_dates, DMP_ID_TO_INCLUDED_MED_DATES, task, post_med_start_vital_dates = [], pre_med_start_vital_dates = [], DMP_ID_to_DATES_to_VITALS_DICT = {}, include_gcsf=False, anc=False):
    post_med_start_date_to_WBC_plus_neutpercent_dict = {date: [DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID][date][WBC_INDEX], DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID][date][NEUTROPHIL_PERCENT_INDEX]] for date in post_med_start_dates if date in DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID]}
    # post_med_start_date_to_WBC_plus_neutpercent_dict = {date: values for date, values in post_med_start_date_to_WBC_plus_neutpercent_dict.items() if 'NA' not in values}
    ## filter out NA/NANs
    post_med_start_date_to_WBC_plus_neutpercent_dict = {date: values for date, values in post_med_start_date_to_WBC_plus_neutpercent_dict.items() if values[0] not in ['NA', 'NAN', None] and not (isinstance(values[0], float) and math.isnan(values[0])) and values[1] not in ['NA', 'NAN', None] and not (isinstance(values[1], float) and math.isnan(values[1]))}

    # DEBUG: Print first few values before conversion
    if len(post_med_start_date_to_WBC_plus_neutpercent_dict) > 0:
        first_date = list(post_med_start_date_to_WBC_plus_neutpercent_dict.keys())[0]
        first_vals = post_med_start_date_to_WBC_plus_neutpercent_dict[first_date]
        # print(f"DEBUG neutropenia_FN: DMP_ID={DMP_ID}, anc={anc}, first_date={first_date}, WBC={first_vals[0]}, Neut_raw={first_vals[1]}")

    # Calculate absolute neutrophil count based on dataset type
    if anc:
        # CAIA dataset: values[1] is already absolute neutrophil count (in K/µL or 10^9/L)
        # Multiply by 1000 to get cells/µL
        post_med_start_date_to_neutrophilcounts_dict = {date: values[1]*1000 for date, values in post_med_start_date_to_WBC_plus_neutpercent_dict.items()}
        # print(f"DEBUG neutropenia_FN: Using ANC mode (CAIA): neutrophil_count = raw_value * 1000")
    else:
        # MSK, MIMICIV_hosp, stanfordEHRSHOT: values[1] is raw percentage (e.g., 60 for 60%)
        # Calculate: WBC (K/µL) * neutrophil_percent / 100 * 1000 = cells/µL
        post_med_start_date_to_neutrophilcounts_dict = {date: values[1]*1000*values[0]/100 for date, values in post_med_start_date_to_WBC_plus_neutpercent_dict.items()}
        # print(f"DEBUG neutropenia_FN: Using percentage mode (MSK/MIMIC/Stanford): neutrophil_count = neutrophil_percent * WBC * 1000 / 100")

    # # DEBUG: Print first converted value
    # if len(post_med_start_date_to_neutrophilcounts_dict) > 0:
    #     first_date = list(post_med_start_date_to_neutrophilcounts_dict.keys())[0]
    #     first_anc = post_med_start_date_to_neutrophilcounts_dict[first_date]
        # print(f"DEBUG neutropenia_FN: first_date={first_date}, calculated_ANC={first_anc} cells/µL")

    post_med_start_date_to_neutrophilcounts_dict = {date: value for date, value in post_med_start_date_to_neutrophilcounts_dict.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
    post_med_start_neutrophil_counts = list(post_med_start_date_to_neutrophilcounts_dict.values())

    baseline_date_to_WBC_plus_neutpercent_dict = {date: [DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID][date][WBC_INDEX], DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID][date][NEUTROPHIL_PERCENT_INDEX]] for date in baseline_judging_dates if date in DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID]}
    # baseline_date_to_WBC_plus_neutpercent_dict = {date: values for date, values in baseline_date_to_WBC_plus_neutpercent_dict.items() if 'NA' not in values}
    ## filter out NA/NANs
    baseline_date_to_WBC_plus_neutpercent_dict = {date: values for date, values in baseline_date_to_WBC_plus_neutpercent_dict.items() if values[0] not in ['NA', 'NAN', None] and not (isinstance(values[0], float) and math.isnan(values[0])) and values[1] not in ['NA', 'NAN', None] and not (isinstance(values[1], float) and math.isnan(values[1]))}

    # # DEBUG: Print first few baseline values before conversion
    # if len(baseline_date_to_WBC_plus_neutpercent_dict) > 0:
    #     first_date = list(baseline_date_to_WBC_plus_neutpercent_dict.keys())[0]
    #     first_vals = baseline_date_to_WBC_plus_neutpercent_dict[first_date]
        # print(f"DEBUG neutropenia_FN baseline: DMP_ID={DMP_ID}, anc={anc}, first_date={first_date}, WBC={first_vals[0]}, Neut_raw={first_vals[1]}")

    # Calculate absolute neutrophil count for baseline based on dataset type
    if anc:
        # CAIA dataset: values[1] is already absolute neutrophil count (in K/µL or 10^9/L)
        baseline_date_to_neutrophilcounts_dict = {date: values[1]*1000 for date, values in baseline_date_to_WBC_plus_neutpercent_dict.items()}
        baseline_neutrophil_counts = [values[1]*1000 for date, values in baseline_date_to_WBC_plus_neutpercent_dict.items()]
        # print(f"DEBUG neutropenia_FN baseline: Using ANC mode (CAIA)")
    else:
        # MSK, MIMICIV_hosp, stanfordEHRSHOT: values[1] is raw percentage (e.g., 60 for 60%)
        baseline_date_to_neutrophilcounts_dict = {date: values[1]*1000*values[0]/100 for date, values in baseline_date_to_WBC_plus_neutpercent_dict.items()}
        baseline_neutrophil_counts = [values[1]*1000*values[0]/100 for date, values in baseline_date_to_WBC_plus_neutpercent_dict.items()]
        # print(f"DEBUG neutropenia_FN baseline: Using percentage mode (MSK/MIMIC/Stanford)")

    # # DEBUG: Print first converted baseline value
    # if len(baseline_date_to_neutrophilcounts_dict) > 0:
    #     first_date = list(baseline_date_to_neutrophilcounts_dict.keys())[0]
    #     first_anc = baseline_date_to_neutrophilcounts_dict[first_date]
        # print(f"DEBUG neutropenia_FN baseline: first_date={first_date}, calculated_ANC={first_anc} cells/µL")

    baseline_date_to_neutrophilcounts_dict = {date: value for date, value in baseline_date_to_neutrophilcounts_dict.items() if value not in ['NA', 'NAN', np.nan, 'nan']}
    baseline_neutrophil_counts = [x for x in baseline_neutrophil_counts if x not in ['NA', 'NAN', np.nan, 'nan']]
    
    if bool(re.match(r'^(febrile_neutropenia)(_next_risk|_risk)?$', task)):
        # febrile neutropenia requires temperature as well
        post_med_start_date_to_temp_dict = {date: DMP_ID_to_DATES_to_VITALS_DICT[DMP_ID][date][Temperature_C_INDEX] for date in post_med_start_vital_dates if date in DMP_ID_to_DATES_to_VITALS_DICT.get(DMP_ID, {})}
        # post_med_start_date_to_temp_dict = {date: float(value) for date, value in post_med_start_date_to_temp_dict.items() if value not in ['NA', 'NAN', np.nan, 'nan']}
        # post_med_start_date_to_temp_dict = {date: value for date, value in post_med_start_date_to_temp_dict.items() if value not in ['NA', 'NAN', np.nan, 'nan'] and pd.notna(value)}
        # filter out NA/NANs
        post_med_start_date_to_temp_dict = {date: value for date, value in post_med_start_date_to_temp_dict.items() if value not in ['NA', 'NAN', np.nan, 'nan', None] and not (isinstance(value, float) and math.isnan(value))}
        post_med_start_date_to_temp_dict = {date: float(value) for date, value in post_med_start_date_to_temp_dict.items() if value not in ['NA', 'NAN', np.nan, 'nan', None]}
        post_med_start_temp_values = list(post_med_start_date_to_temp_dict.values())
        baseline_date_to_temp_dict = {date: DMP_ID_to_DATES_to_VITALS_DICT[DMP_ID][date][Temperature_C_INDEX] for date in pre_med_start_vital_dates if date in DMP_ID_to_DATES_to_VITALS_DICT.get(DMP_ID, {})}
        # baseline_date_to_temp_dict = {date: float(value) for date, value in baseline_date_to_temp_dict.items() if value not in ['NA', 'NAN', np.nan, 'nan']}
        # baseline_date_to_temp_dict = {date: value for date, value in baseline_date_to_temp_dict.items() if value not in ['NA', 'NAN', np.nan, 'nan'] and pd.notna(value)}
        # filter out NA/NANs
        baseline_date_to_temp_dict = {date: value for date, value in baseline_date_to_temp_dict.items() if value not in ['NA', 'NAN', np.nan, 'nan', None] and not (isinstance(value, float) and math.isnan(value))}
        baseline_date_temp_values = list(baseline_date_to_temp_dict.values())
        
        # get date combos that provide FN reference (neut count + temp within 2-days of each other)
        post_med_start_date_to_neutrophilcounts_and_temp = match_neutrophil_and_temp(
            post_med_start_date_to_neutrophilcounts_dict, post_med_start_date_to_temp_dict, days=2
        )
        baseline_date_to_neutrophilcounts_and_temp = match_neutrophil_and_temp(
            baseline_date_to_neutrophilcounts_dict, baseline_date_to_temp_dict, days=2
        )
        post_med_start_date_to_FN_status_dict = define_FN(post_med_start_date_to_neutrophilcounts_and_temp)
        baseline_date_to_FN_status_dict = define_FN(baseline_date_to_neutrophilcounts_and_temp)
    
    if 'febrile' not in task:
        ## 2. check for lack of 2 relevant measurements prior to med start
        if len(baseline_neutrophil_counts) < 2:
            removal_dict["lack_2_rel_past_removal"] = True
            return {0: (removal_dict, [], '', '', '', '')}
            
        ## 3. check for baseline side effect
        if min(baseline_neutrophil_counts) < 1000:
            removal_dict["baseline_side_effect_removal"] = True
            return {0: (removal_dict, [], '', '', '', '')}
        else:
            if len(post_med_start_neutrophil_counts) == 0:
                removal_dict["lack_1_rel_future_removal"] = True
                return {0: (removal_dict, [], '', '', '', '')}

            # whole med period measurements (for entire_y)
            whole_med_period_neutrophil_measurements = {}
            for inf_idx, inf_date in enumerate(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]):
                inf_window_neutrophil_measurements = {date: value for date, value in post_med_start_date_to_neutrophilcounts_dict.items() if date >= inf_date and (inf_idx == len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 or date < min(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx+1], DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx]+30))}
                whole_med_period_neutrophil_measurements.update(inf_window_neutrophil_measurements)
            if len(whole_med_period_neutrophil_measurements) > 0 and min(whole_med_period_neutrophil_measurements.values()) < 1000:
                entire_y = 1
            else:
                entire_y = 0

            # PASS 1: Find when neutropenia occurs (which idx has y==1)
            neut_idx = None
            temp_y_list = []
            for idx, med_start_date in enumerate(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]):
                next_infusion_date = DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][idx+1] if idx < len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 else float('inf')
                end_window = min(next_infusion_date, med_start_date + 30)

                current_neutrophil_measurements = {date: value for date, value in post_med_start_date_to_neutrophilcounts_dict.items() if date >= med_start_date and date < end_window}

                if len(current_neutrophil_measurements) > 0 and min(current_neutrophil_measurements.values()) < 1000:
                    y = 1
                    neut_idx = idx  # Record where neutropenia happens
                    earliest_neutropenia_date = min(date for date, value in current_neutrophil_measurements.items() if value < 1000)
                else:
                    y = 0
                    earliest_neutropenia_date = None

                temp_y_list.append((y, earliest_neutropenia_date, current_neutrophil_measurements))

                # # Print ground truth determination for neutropenia task
                # print(f"Neutropenia GT: DMP_ID={DMP_ID}, infusion_idx={idx}, y={y}, "
                #       f"min_neut_count={min(current_neutrophil_measurements.values()) if current_neutrophil_measurements else 'N/A'}, "
                #       f"earliest_neut_date={earliest_neutropenia_date}, "
                #       f"num_measurements={len(current_neutrophil_measurements)}")

                if y == 1:
                    break  # Early termination when neutropenia found

            # PASS 2: Build results with correct time_to_event
            results = {}
            for idx, med_start_date in enumerate(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]):
                # 1. check if the last contact date is outside of the washout period, if so censor and break out the for loop
                if idx == len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 and DMP_IDs_to_lastcontactdate_dict[DMP_ID] < med_start_date + 30:
                    removal_dict_instance = removal_dict.copy()
                    removal_dict_instance["censor_removal"] = True
                    results[idx] = (removal_dict_instance, [], '', '', entire_y, '')
                    return results

                # Retrieve y from pass 1
                if idx >= len(temp_y_list):
                    break
                y, earliest_neutropenia_date, current_neutrophil_measurements = temp_y_list[idx]

                if include_gcsf:
                    if y == 1:
                        y = (y, earliest_neutropenia_date)
                    else:
                        y = (y, None)

                # Calculate correct time_to_event (inclusive of current infusion)
                if neut_idx is not None:
                    # Neutropenia occurs - count cycles from current idx to neut_idx (inclusive)
                    current_num_durations_to_FN_or_cutoff = (neut_idx - idx) + 1
                else:
                    # No neutropenia - count remaining cycles (censored, inclusive)
                    current_num_durations_to_FN_or_cutoff = (len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]) - idx) + 1

                input_data_dates = [date for date in all_dates if date <= med_start_date]
                results[idx] = (removal_dict.copy(), input_data_dates, med_start_date, y, entire_y, current_num_durations_to_FN_or_cutoff)

                if (not include_gcsf and y == 1) or (include_gcsf and y[0] == 1):
                    return results

            return results
            
    else: ## febrile neutropenia
        ## 2. check for lack of 2 relevant measurements prior to med start
        if len(baseline_date_to_neutrophilcounts_and_temp) < 2:
            removal_dict["lack_2_rel_past_removal"] = True
            return {0: (removal_dict, [], '', '', '', '')}
            
        ## 3. check for baseline side effect
        if baseline_date_to_FN_status_dict[0]:
            # print("DMP_ID with baseline febrile neutropenia:", DMP_ID, baseline_date_to_neutrophilcounts_and_temp)
            removal_dict["baseline_side_effect_removal"] = True
            return {0: (removal_dict, [], '', '', '', '')}
        else:
            if len(post_med_start_date_to_neutrophilcounts_and_temp) == 0:
                removal_dict["lack_1_rel_future_removal"] = True
                return {0: (removal_dict, [], '', '', '', '')}
                
            # whole med period measurements (for entire_y)
            whole_med_period_neutrophil_measurements = {}
            whole_med_period_temp_measurements = {}
            for inf_idx, inf_date in enumerate(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]):
                inf_window_neutrophil_measurements = {date: value for date, value in post_med_start_date_to_neutrophilcounts_dict.items() if date >= inf_date and (inf_idx == len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 or date < min(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx+1], DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx]+30))}
                inf_window_temp_measurements = {date: value for date, value in post_med_start_date_to_temp_dict.items() if date >= inf_date and (inf_idx == len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 or date < min(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx+1], DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx]+30))}
                whole_med_period_neutrophil_measurements.update(inf_window_neutrophil_measurements)
                whole_med_period_temp_measurements.update(inf_window_temp_measurements)
            whole_med_period_neutrophil_and_temp = match_neutrophil_and_temp(
                whole_med_period_neutrophil_measurements, whole_med_period_temp_measurements, days=2
            )
            whole_med_period_FN_status = define_FN(whole_med_period_neutrophil_and_temp)
            if whole_med_period_FN_status[0]:
                entire_y = 1
            else:
                entire_y = 0
            # PASS 1: Find when FN occurs (which idx has y==1)
            fn_idx = None
            temp_y_list = []
            for idx, med_start_date in enumerate(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]):
                next_infusion_date = DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][idx+1] if idx < len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 else float('inf')
                end_window = min(next_infusion_date, med_start_date + 30)

                current_neutrophil_measurements = {date: value for date, value in post_med_start_date_to_neutrophilcounts_dict.items() if date >= med_start_date and date < end_window}
                current_temp_measurements = {date: value for date, value in post_med_start_date_to_temp_dict.items() if date >= med_start_date and date < end_window}

                current_neutrophil_and_temp = match_neutrophil_and_temp(
                    current_neutrophil_measurements, current_temp_measurements, days=2
                )
                current_FN_status = define_FN(current_neutrophil_and_temp)

                if current_FN_status[0]:
                    y = 1
                    fn_idx = idx  # Record where FN happens
                else:
                    y = 0

                temp_y_list.append((y, current_FN_status))

                if y == 1:
                    break  # Early termination when FN found

            # PASS 2: Build results with correct time_to_event
            results = {}
            for idx, med_start_date in enumerate(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]):
                # 1. check if the last contact date is outside of the washout period, if so censor and break out the for loop
                if idx == len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 and DMP_IDs_to_lastcontactdate_dict[DMP_ID] < med_start_date + 30:
                    removal_dict_instance = removal_dict.copy()
                    removal_dict_instance["censor_removal"] = True
                    results[idx] = (removal_dict_instance, [], '', '', entire_y, '')
                    return results

                # Retrieve y from pass 1
                if idx >= len(temp_y_list):
                    break
                y, current_FN_status = temp_y_list[idx]

                if include_gcsf:
                    if y == 1:
                        earliest_FN_date = current_FN_status[1]
                        y = (y, earliest_FN_date)
                    else:
                        y = (y, None)

                # Calculate correct time_to_event (inclusive of current infusion)
                if fn_idx is not None:
                    # FN occurs - count cycles from current idx to FN idx (inclusive)
                    current_num_durations_to_FN_or_cutoff = (fn_idx - idx) + 1
                else:
                    # No FN - count remaining cycles (censored, inclusive)
                    current_num_durations_to_FN_or_cutoff = (len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]) - idx) + 1

                input_data_dates = [date for date in all_dates if date <= med_start_date]
                results[idx] = (removal_dict.copy(), input_data_dates, med_start_date, y, entire_y, current_num_durations_to_FN_or_cutoff)

                if (not include_gcsf and y == 1) or (include_gcsf and y[0] == 1):
                    return results

            return results

def rolling_c2_within(baseline_dict, window_days=7, baseline_strategy='last', floor=0.5):
    """
    Detects KDIGO c2 within a single time series: for each value at t,
    compare against a baseline from the prior window_days.
    Returns (status_bool, earliest_date or None).
    Assumes integer day keys; if using datetimes, replace arithmetic with .days.
    """
    items = sorted(baseline_dict.items())  # [(day, scr)] ascending
    prev = []  # prior values we've seen

    for t, v in items:
        window = [(d, val) for d, val in prev if (t - window_days) <= d < t]
        if not window:
            prev.append((t, v))
            continue

        if baseline_strategy == 'last':
            ref = max(window, key=lambda x: x[0])[1]
        elif baseline_strategy == 'min':
            ref = min(val for _, val in window)
        elif baseline_strategy == 'median2':
            days = sorted({d for d, _ in window})
            last_days = days[-2:] if len(days) >= 2 else days[-1:]
            vals = []
            for day in last_days:
                same_day_vals = [val for d, val in window if d == day]
                vals.append(same_day_vals[-1])  # last draw of each day
            vals.sort()
            ref = vals[len(vals)//2] if len(vals) % 2 == 1 else (vals[0] + vals[1]) / 2
        else:
            ref = max(window, key=lambda x: x[0])[1]

        ref = max(ref, floor)  # protect against tiny baselines

        if v >= 1.5 * ref:
            return True, t

        prev.append((t, v))

    return False, None

def rolling_c2(baseline_dict, post_dict, window_days=7, baseline_strategy='last', floor=0.5):
    """
    baseline_dict: {day: scr} for pre-med (sorted ascending by day)
    post_dict: {day: scr} for post-med (sorted ascending by day)
    window_days: size of lookback window for baseline (KDIGO: 7)
    baseline_strategy:
    'last' -> use the most recent prior SCr within the window
    'min' -> use the minimum SCr within the window (more sensitive)
    'median2' -> median of SCr on the last up to 2 distinct prior days (robust)
    floor: minimum baseline allowed (mg/dL) to avoid tiny baselines
    Returns: (status_bool, earliest_c2_date or None)
    """
    # ensure sorted dicts
    baseline_items = sorted(baseline_dict.items())
    post_items = sorted(post_dict.items())

    # "prev" holds all values available before evaluating a post point (starts with pre values)
    prev = list(baseline_items)
    c2_date = None

    for t, v in post_items:
        # collect prior values within [t - window_days, t)
        window = [(d, val) for d, val in prev if (t - window_days) <= d < t]
        if not window:
            # nothing to compare in last 7 days; cannot call c2 here
            prev.append((t, v))
            continue

        if baseline_strategy == 'last':
            # most recent prior value
            ref = max(window, key=lambda x: x[0])[1]
        elif baseline_strategy == 'min':
            ref = min(val for _, val in window)
        elif baseline_strategy == 'median2':
            # median of values on the last up to 2 distinct days
            days = sorted({d for d, _ in window})
            last_days = days[-2:] if len(days) >= 2 else days[-1:]
            vals = []
            for day in last_days:
                # if multiple draws per day, use the last draw that day
                same_day_vals = [val for d, val in window if d == day]
                vals.append(same_day_vals[-1])
            vals.sort()
            ref = vals[len(vals)//2] if len(vals) % 2 == 1 else (vals[0] + vals[1]) / 2
        else:
            # default to 'last'
            ref = max(window, key=lambda x: x[0])[1]

        ref = max(ref, floor) # protect against unrealistically low baselines (e.g., lab artifact)

        if v >= 1.5 * ref:
            return True, t

        # make this value available for later windows
        prev.append((t, v))

    return False, None

def AKI_adjudication(DMP_ID, removal_dict, pre_med_start_dates, post_med_start_dates, baseline_judging_dates, DMP_ID_to_DATES_to_TESTSVALUES_DICT, MED_DMP_ID_TO_START_END_DATE_DICT, DMP_IDs_to_lastcontactdate_dict, all_dates, DMP_ID_TO_INCLUDED_MED_DATES, task):
    baseline_date_to_creatinine_levels = {date: DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID][date][CREATININE_INDEX] for date in baseline_judging_dates if date in DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID]}
    baseline_date_to_creatinine_levels = {date: value for date, value in baseline_date_to_creatinine_levels.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
    baseline_creatinine_levels = list(baseline_date_to_creatinine_levels.values())
    post_med_start_date_to_creatinine_levels = {date: DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID][date][CREATININE_INDEX] for date in post_med_start_dates if date in DMP_ID_to_DATES_to_TESTSVALUES_DICT[DMP_ID]}
    post_med_start_date_to_creatinine_levels = {date: value for date, value in post_med_start_date_to_creatinine_levels.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
    post_med_start_creatinine_levels = list(post_med_start_date_to_creatinine_levels.values())

    # sort
    baseline_date_to_creatinine_levels = dict(sorted(baseline_date_to_creatinine_levels.items()))
    post_med_start_date_to_creatinine_levels = dict(sorted(post_med_start_date_to_creatinine_levels.items()))

    ## 2. check for lack of 2 relevant measurements prior to med start
    if len(baseline_creatinine_levels) < 2:
        removal_dict["lack_2_rel_past_removal"] = True
        return {0: (removal_dict, [], '', '', '', '')}
        
    ## 3. check for baseline side effect
    ### 3.1 2-day increase of 0.3
    baseline_c1_AKI_status = any(
                        late_value - early_value >= 0.3
                        for early_date, late_date in combinations(baseline_date_to_creatinine_levels.keys(), 2)
                        if late_date <= early_date + 2 and late_date - early_date >= 0
                        for early_value, late_value in [(baseline_date_to_creatinine_levels[early_date], baseline_date_to_creatinine_levels[late_date])]
                    )
    if baseline_c1_AKI_status:
        # print("DMP_ID with baseline c1:", DMP_ID, baseline_date_to_creatinine_levels)
        removal_dict["baseline_side_effect_removal"] = True
        return {0: (removal_dict, [], '', '', '', '')}
    else:
        ### 3.2 1.5X baseline (first ever measurement)
        baseline_c2_AKI_status, baseline_c2_AKI_date = rolling_c2_within(
            baseline_date_to_creatinine_levels,
            window_days=7,
            baseline_strategy='last',   # or 'median2' for robustness
            floor=0.5
        )
        if baseline_c2_AKI_status:
            # print("DMP_ID with baseline c2:", DMP_ID, baseline_date_to_creatinine_levels, baseline_c2_AKI_date)
            removal_dict["baseline_side_effect_removal"] = True
            return {0: (removal_dict, [], '', '', '', '')}
        else:
            if len(post_med_start_creatinine_levels) == 0:
                removal_dict["lack_1_rel_future_removal"] = True
                return {0: (removal_dict, [], '', '', '', '')}

            ## whole med period measurements (for entire_y)
            whole_med_period_creatinine_measurements = {}
            for inf_idx, inf_date in enumerate(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]):
                inf_window_measurements = {date: value for date, value in post_med_start_date_to_creatinine_levels.items() if date >= inf_date and (inf_idx == len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 or date < min(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx+1], DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx]+30))}
                whole_med_period_creatinine_measurements.update(inf_window_measurements)
            whole_med_period_AKI_status_c1 = False
            whole_med_period_AKI_status_c2 = False
            # if len(whole_med_period_creatinine_measurements) > 0:
            ### criterion 1: Increase in SCr to ⩾0.3 mg/dl within 48 hours (include the last one prior to med start)
            last_baseline_date = max(baseline_date_to_creatinine_levels.keys())
            whole_med_period_measurements_with_last = {last_baseline_date: baseline_date_to_creatinine_levels[last_baseline_date]}
            whole_med_period_measurements_with_last.update(whole_med_period_creatinine_measurements)

            # TODO: check - added 121825 - remove NANs
            whole_med_period_creatinine_measurements = {date: value for date, value in whole_med_period_creatinine_measurements.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
            whole_med_period_measurements_with_last = {date: value for date, value in whole_med_period_measurements_with_last.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
            
            whole_med_period_AKI_status_c1 = any(
                late_value - early_value >= 0.3
                for early_date, late_date in combinations(whole_med_period_measurements_with_last.keys(), 2)
                if late_date <= early_date + 2 and late_date - early_date >= 0
                for early_value, late_value in [(whole_med_period_measurements_with_last[early_date], whole_med_period_measurements_with_last[late_date])]
            )
            
            # criterion 2: 1.5X baseline using rolling window
            whole_med_period_AKI_status_c2, _ = rolling_c2(
                baseline_date_to_creatinine_levels,
                whole_med_period_creatinine_measurements,
                window_days=7,
                baseline_strategy='last',
                floor=0.5
            )
            if whole_med_period_AKI_status_c1 or whole_med_period_AKI_status_c2:
                entire_y = 1
            else:
                entire_y = 0
            # PASS 1: Find when AKI occurs (which idx has y==1)
            aki_idx = None
            temp_y_list = []

            for idx, med_start_date in enumerate(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]):
                next_infusion_date = DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][idx+1] if idx < len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 else float('inf')
                end_window = min(next_infusion_date, med_start_date + 30)

                current_creatinine_measurements = {date: value for date, value in post_med_start_date_to_creatinine_levels.items() if date >= med_start_date and date < end_window}

                y = 0
                if len(current_creatinine_measurements) > 0:
                    last_baseline_date = max(baseline_date_to_creatinine_levels.keys())
                    current_measurements_with_last = {last_baseline_date: baseline_date_to_creatinine_levels[last_baseline_date]}
                    current_measurements_with_last.update(current_creatinine_measurements)

                    current_creatinine_measurements = {date: value for date, value in current_creatinine_measurements.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
                    current_measurements_with_last = {date: value for date, value in current_measurements_with_last.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}

                    current_c1_AKI_status = any(
                        late_value - early_value >= 0.3
                        for early_date, late_date in combinations(current_measurements_with_last.keys(), 2)
                        if late_date <= early_date + 2 and late_date - early_date >= 0
                        for early_value, late_value in [(current_measurements_with_last[early_date], current_measurements_with_last[late_date])]
                    )

                    current_c2_AKI_status, _ = rolling_c2(
                        baseline_date_to_creatinine_levels,
                        current_creatinine_measurements,
                        window_days=7,
                        baseline_strategy='last',
                        floor=0.5
                    )

                    if current_c1_AKI_status or current_c2_AKI_status:
                        y = 1
                        aki_idx = idx

                temp_y_list.append(y)

                if y == 1:
                    break

            # PASS 2: Build results with correct time_to_event
            results = {}
            for idx, med_start_date in enumerate(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]):
                # 1. check if the last contact date is outside of the washout period, if so censor and break out the for loop
                if idx == len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 and DMP_IDs_to_lastcontactdate_dict[DMP_ID] < med_start_date + 30:
                    removal_dict_instance = removal_dict.copy()
                    removal_dict_instance["censor_removal"] = True
                    results[idx] = (removal_dict_instance, [], '', '', entire_y, '')
                    return results

                # Retrieve y from pass 1
                if idx >= len(temp_y_list):
                    break
                y = temp_y_list[idx]

                # Calculate correct time_to_event (inclusive of current infusion)
                if aki_idx is not None:
                    current_num_durations_to_AKI_or_cutoff = (aki_idx - idx) + 1
                else:
                    current_num_durations_to_AKI_or_cutoff = (len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]) - idx) + 1

                # Re-fetch measurements for detailed checking (if needed for logging)
                next_infusion_date = DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][idx+1] if idx < len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 else float('inf')
                end_window = min(next_infusion_date, med_start_date + 30)
                current_creatinine_measurements = {date: value for date, value in post_med_start_date_to_creatinine_levels.items() if date >= med_start_date and date < end_window}

                aki_date = None
                if len(current_creatinine_measurements) > 0 and y == 1:
                    ### criterion 1: Increase in SCr to ⩾0.3 mg/dl within 48 hours (include the last one prior to med start)
                    last_baseline_date = max(baseline_date_to_creatinine_levels.keys())
                    current_measurements_with_last = {last_baseline_date: baseline_date_to_creatinine_levels[last_baseline_date]}
                    current_measurements_with_last.update(current_creatinine_measurements)
                    
                    # TODO: check - added 121825 - remove NANs
                    current_creatinine_measurements = {date: value for date, value in current_creatinine_measurements.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
                    current_measurements_with_last = {date: value for date, value in current_measurements_with_last.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}

                    current_c1_AKI_status = any(
                        late_value - early_value >= 0.3
                        for early_date, late_date in combinations(current_measurements_with_last.keys(), 2)
                        if late_date <= early_date + 2 and late_date - early_date >= 0
                        for early_value, late_value in [(current_measurements_with_last[early_date], current_measurements_with_last[late_date])]
                    )
                    
                    # whole med period measurements (for entire_y)
                    whole_med_period_creatinine_measurements = {}
                    for inf_idx, inf_date in enumerate(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID]):
                        inf_window_measurements = {date: value for date, value in post_med_start_date_to_creatinine_levels.items() if date >= inf_date and (inf_idx == len(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID])-1 or date < min(DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx+1], DMP_ID_TO_INCLUDED_MED_DATES[DMP_ID][inf_idx]+30))}
                        whole_med_period_creatinine_measurements.update(inf_window_measurements)

                    entire_c1_AKI_status = False
                    entire_c2_AKI_status = False
                    if len(whole_med_period_creatinine_measurements) > 0:
                        ### criterion 1: Increase in SCr to ⩾0.3 mg/dl within 48 hours (include the last one prior to med start)
                        last_baseline_date = max(baseline_date_to_creatinine_levels.keys())
                        whole_med_period_measurements_with_last = {last_baseline_date: baseline_date_to_creatinine_levels[last_baseline_date]}
                        whole_med_period_measurements_with_last.update(whole_med_period_creatinine_measurements)
                        
                        # TODO: check - added 121825 - remove NANs
                        whole_med_period_creatinine_measurements = {date: value for date, value in whole_med_period_creatinine_measurements.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))}
                        whole_med_period_measurements_with_last = {date: value for date, value in whole_med_period_measurements_with_last.items() if value not in ['NA', 'NAN', None] and not (isinstance(value, float) and math.isnan(value))} 

                        entire_c1_AKI_status = any(
                            late_value - early_value >= 0.3
                            for early_date, late_date in combinations(whole_med_period_measurements_with_last.keys(), 2)
                            if late_date <= early_date + 2 and late_date - early_date >= 0
                            for early_value, late_value in [(whole_med_period_measurements_with_last[early_date], whole_med_period_measurements_with_last[late_date])]
                        )
                        
                        # criterion 2: 1.5X baseline using rolling window
                        entire_c2_AKI_status, _ = rolling_c2(
                            baseline_date_to_creatinine_levels,
                            whole_med_period_creatinine_measurements,
                            window_days=7,
                            baseline_strategy='last',
                            floor=0.5
                        )
                        
                    c1_aki_date = None
                    if current_c1_AKI_status:
                        c1_aki_date = min(
                                    late_date
                                    for early_date, late_date in combinations(current_measurements_with_last.keys(), 2)
                                    if late_date <= early_date + 2 and late_date - early_date >= 0
                                    for early_value, late_value in [(current_measurements_with_last[early_date], current_measurements_with_last[late_date])]
                                    if late_value - early_value >= 0.3
                                )

                    ### criterion 2: 1.5X baseline using rolling window
                    current_c2_AKI_status, c2_aki_date = rolling_c2(
                        baseline_date_to_creatinine_levels,
                        current_creatinine_measurements,
                        window_days=7,
                        baseline_strategy='last',
                        floor=0.5
                    )

                    if current_c1_AKI_status or current_c2_AKI_status:
                        if current_c1_AKI_status and current_c2_AKI_status:
                            aki_date = min(c1_aki_date, c2_aki_date)
                        elif current_c1_AKI_status:
                            aki_date = c1_aki_date
                        elif current_c2_AKI_status:
                            aki_date = c2_aki_date

                input_data_dates = [date for date in all_dates if date <= med_start_date]
                results[idx] = (removal_dict.copy(), input_data_dates, med_start_date, y, entire_y, current_num_durations_to_AKI_or_cutoff)
                
                if y == 1:
                    return results
                    
            return results