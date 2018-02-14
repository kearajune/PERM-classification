
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt


# In[2]:

data = pd.read_csv("us_perm_visas.csv",low_memory=False)


# In[3]:

new_data = data


# In[4]:

def drop_columns(data):
    '''Drop unnecessary columns.
    Parameters:
        data_frame (pd.Dataframe): the dataset being edited.
    Returns:
        A pd.DataFrame object with the above elements changed.
    '''
    # Add the columns to delete in a list. 
    drop_list = ['add_these_pw_job_title_9089',
        'foreign_worker_info_alt_edu_experience',
        'foreign_worker_info_birth_country','foreign_worker_info_postal_code',
        'foreign_worker_info_rel_occup_exp','foreign_worker_info_req_experience',
        'foreign_worker_info_training_comp','foreign_worker_ownership_interest',
        'fw_info_alt_edu_experience','fw_info_birth_country',
        'fw_info_postal_code','fw_info_rel_occup_exp','fw_info_req_experience',
        'fw_info_training_comp','fw_ownership_interest',
        'ji_foreign_worker_live_on_premises','ji_fw_live_on_premises',
        'ji_offered_to_sec_j_foreign_worker','ji_offered_to_sec_j_fw',
        'job_info_alt_cmb_ed_oth_yrs','job_info_alt_combo_ed',
        'job_info_alt_combo_ed_other','job_info_alt_field_name',
        'job_info_alt_occ','job_info_alt_occ_job_title',
        'job_info_alt_occ_num_months','job_info_training_field',
        'job_info_training_num_months','orig_file_date','orig_case_no',
        'recr_info_job_fair_to','recr_info_job_fair_from',
        'recr_info_on_campus_recr_to','recr_info_on_campus_recr_from',
        'ri_coll_teach_select_date','ri_coll_tch_basic_process',
        'recr_info_coll_teach_comp_proc','recr_info_pro_org_advert_to',
        'recr_info_pro_org_advert_from' ,'recr_info_prof_org_advert_to',
        'recr_info_prof_org_advert_from','pw_source_name_other_9089',
        'ri_pvt_employment_firm_to','ri_pvt_employment_firm_from',
        'ri_us_workers_considered','recr_info_radio_tv_ad_from',
        'recr_info_radio_tv_ad_to','ri_campus_placement_to',
        'ri_campus_placement_from','ri_employee_referral_prog_from',
        'ri_employee_referral_prog_to','pw_job_title_908',
        'recr_info_barg_rep_notified','ri_coll_teach_pro_jnl',
        'ri_job_search_website_to','ri_job_search_website_from',
        'preparer_info_title','pw_job_title_9089', 'recr_info_second_ad_start']
    drop_list2 =  ['agent_city','agent_state','employer_address_1',
        'employer_address_2','employer_country','employer_phone',
        'employer_phone_ext','employer_postal_code','employer_city',
        'foreign_worker_info_city','employer_decl_info_title',
        'foreign_worker_info_inst','foreign_worker_info_state',
        'foreign_worker_info_major','job_info_alt_combo_ed_exp',
        'job_info_alt_field','job_info_combo_occupation',
        'job_info_job_req_normal','job_info_major','job_info_training',
        'job_info_job_title','job_info_work_city','job_info_work_postal_code',
        'naics_2007_us_code','naics_code','naics_us_code', 'naics_us_code_2007',
        'preparer_info_emp_completed','pw_determ_date', 'pw_expire_date',
        'pw_level_9089', 'pw_soc_code','pw_soc_title', 'pw_source_name_9089',
        'pw_track_num', 'rec_info_barg_rep_notified','recr_info_first_ad_start',
        'recr_info_sunday_newspaper','recr_info_swa_job_order_end',
        'recr_info_swa_job_order_start','ri_1st_ad_newspaper_name',
        'ri_2nd_ad_newspaper_name','ri_2nd_ad_newspaper_or_journal',
        'ri_employer_web_post_from', 'ri_employer_web_post_to',
        'ji_live_in_dom_svc_contract','ri_local_ethnic_paper_from',
        'ri_local_ethnic_paper_to', "job_info_work_state",
        'ri_posted_notice_at_worksite','schd_a_sheepherder', 'us_economic_sector',
        'wage_offer_from_9089','wage_offered_from_9089','wage_offer_to_9089',
        'wage_offered_to_9089','employer_name','wage_offer_unit_of_pay_9089',
        'wage_offered_unit_of_pay_9089','case_received_date','application_type',
        'job_info_education','job_info_education_other']
    # Drop the columns. 
    new_data = data.drop(drop_list+drop_list2,axis=1)
    return(new_data)


# In[5]:

def merge_columns(new_data):
    '''Merge duplicate columns.
    Parameters:
        data_frame (pd.Dataframe): the dataset being edited.
    Returns:
        A pd.DataFrame object with the above elements changed.
    '''
    for a,b in [["case_number","case_no"],
        ["country_of_citizenship","country_of_citzenship"],
        ["foreign_worker_info_education_other","fw_info_education_other"],
        ["naics_title","naics_2007_us_title"],
        ["naics_us_title","naics_us_title_2007"]]:
    
        new_data[a] = new_data[[a,b]].fillna('').sum(axis=1)
        new_data = new_data.drop([b],axis=1)
    
    new_data["naics_title"] = new_data[["naics_title",
                                "naics_us_title"]].fillna('').sum(axis=1)
    new_data = new_data.drop(["naics_us_title"],axis=1)
    
    new_data["foreign_worker_yr_rel_edu_completed"] =                         new_data[["fw_info_yr_rel_edu_completed",
                        "foreign_worker_yr_rel_edu_completed"]].fillna(0)
    new_data = new_data.drop(["fw_info_yr_rel_edu_completed"],axis=1)
    new_data["foreign_worker_yr_rel_edu_completed"] =         new_data["foreign_worker_yr_rel_edu_completed"].replace(0, np.nan)
    
    return(new_data)


# In[6]:

def standardize_column_info(new_data):
    '''Takes a data frame and standardizes `pw_unit_of_pay` and
    `employer_state', turns `decision_date`, 
    `foreign_worker_yr_rel_edu_completed` and `employer_yr_estab` into
    datetime objects, turns `pw_amount_9089` into a float, and turns 
    `recr_info_coll_univ_teacher`, `recr_info_employer_rec_payment`,
    `recr_info_professional_occ`, `refile`, `ri_layoff_in_past_six_months`,
    `ji_live_in_domestic_service`, `job_info_foreign_ed`,
    `job_info_foreign_lang_req` and `job_info_experience` into booleans,
    and combines duplicate case numbers.
    Parameters:
        data_frame (pd.Dataframe): the dataset being edited.
    Returns:
        A pd.DataFrame object with the above elements changed.
    '''
    # Uniforming the pw_unit_of_pay_9089 rows. 
    new_data["pw_unit_of_pay_9089"] =                     new_data["pw_unit_of_pay_9089"].replace("yr", "Year")
    new_data["pw_unit_of_pay_9089"] =                     new_data["pw_unit_of_pay_9089"].replace("mth", "Month")
    new_data["pw_unit_of_pay_9089"] =                     new_data["pw_unit_of_pay_9089"].replace("bi", "Bi-Weekly")
    new_data["pw_unit_of_pay_9089"] =                     new_data["pw_unit_of_pay_9089"].replace("hr", "Hour")
    new_data["pw_unit_of_pay_9089"] =                     new_data["pw_unit_of_pay_9089"].replace("wk", "Week")
    
    # Creating state to abbreviation Dictionary. 
    us_state_abbrev = {'Alabama': 'AL','Alaska': 'AK','Arizona': 'AZ',
        'Arkansas': 'AR','California': 'CA', 'Colorado': 'CO',
        'Connecticut': 'CT','Delaware': 'DE','Florida': 'FL','Georgia': 'GA',
        'Hawaii': 'HI','Idaho': 'ID','Illinois': 'IL','Indiana': 'IN',
        'Iowa': 'IA','Kansas': 'KS','Kentucky': 'KY','Louisiana': 'LA',
        'Maine': 'ME','Maryland': 'MD','Massachusetts': 'MA','Michigan': 'MI',
        'Minnesota': 'MN','Mississippi': 'MS','Missouri': 'MO',
        'Montana': 'MT','Nebraska': 'NE','Nevada': 'NV','New Hampshire': 'NH',
        'New Jersey': 'NJ','New Mexico': 'NM','New York': 'NY',
        'North Carolina': 'NC','North Dakota': 'ND','Ohio': 'OH',
        'Oklahoma': 'OK','Oregon': 'OR','Pennsylvania': 'PA',
        'Rhode Island': 'RI','South Carolina': 'SC','South Dakota': 'SD',
        'Tennessee': 'TN','Texas': 'TX','Utah': 'UT','Vermont': 'VT',
        'Virginia': 'VA','Washington': 'WA','West Virginia': 'WV',
        'Wisconsin': 'WI','Wyoming': 'WY','DISTRICT OF COLUMBIA': 'DC',
        'VIRGIN ISLANDS' : 'VI','BRITISH COLUMBIA' :'BC','PUERTO RICO': 'PR',
        'MARSHALL ISLANDS':'MH','NORTHERN MARIANA ISLANDS':'MP','GUAM':'GU'}
    us_state_abbrev = {state.upper(): abrev for state, 
                       abrev in us_state_abbrev.items()}

    # Convert to abbreviations.
    new_data.loc[new_data["employer_state"].str.len()>2,"employer_state"]=                new_data[new_data["employer_state"].str.len()>2]                ["employer_state"].replace(us_state_abbrev)
    
    # Turning decision_date into datetime object. 
    new_data['decision_date'] = pd.to_datetime(new_data['decision_date'],
                                               format = "%Y-%m-%d")

    # Turn foreign_worker_yr_rel_edu_completed into datetime object.
    mask1 = new_data['foreign_worker_yr_rel_edu_completed'] < 1900
    mask2 = new_data['foreign_worker_yr_rel_edu_completed'].isnull()
    new_data.loc[mask1,'foreign_worker_yr_rel_edu_completed'] =             new_data.loc[mask1,'foreign_worker_yr_rel_edu_completed'] = 0
    new_data.loc[mask1,'foreign_worker_yr_rel_edu_completed'] =             new_data.loc[mask2,'foreign_worker_yr_rel_edu_completed'] = 0

    new_data.foreign_worker_yr_rel_edu_completed =                 new_data.foreign_worker_yr_rel_edu_completed.astype(int)

    mask3 = new_data['foreign_worker_yr_rel_edu_completed'] != 0
    new_data.loc[mask3,'foreign_worker_yr_rel_edu_completed'] =                     pd.to_datetime(new_data.loc[mask3,
                    'foreign_worker_yr_rel_edu_completed'],format='%Y')

    a = pd.to_datetime(1970,format="%Y")
    new_data['foreign_worker_yr_rel_edu_completed'] =         new_data['foreign_worker_yr_rel_edu_completed'].replace(a,pd.NaT)
    
    # Turn employer_yr_estab into datetime object. 
    mask1 = new_data['employer_yr_estab'] < 1900
    mask2 = new_data['employer_yr_estab'].isnull()
    new_data.loc[mask1,'employer_yr_estab'] =                             new_data.loc[mask1,'employer_yr_estab'] = 0
    new_data.loc[mask1,'employer_yr_estab'] =                             new_data.loc[mask2,'employer_yr_estab'] = 0

    new_data.employer_yr_estab = new_data.employer_yr_estab.astype(int)

    mask3 = new_data['employer_yr_estab'] != 0
    new_data.loc[mask3,'employer_yr_estab'] =         pd.to_datetime(new_data.loc[mask3,'employer_yr_estab'],format='%Y')

    a = pd.to_datetime(1970,format="%Y")
    new_data['employer_yr_estab'] =                             new_data['employer_yr_estab'].replace(a,pd.NaT)

    # Turning into float. 
    new_data['pw_amount_9089'] =                 pd.to_numeric(new_data['pw_amount_9089'].str.replace(",",""))

    # Turning Yes or into booleans. 
    for column in ['recr_info_coll_univ_teacher',
        'recr_info_employer_rec_payment','recr_info_professional_occ',
                   'refile','ri_layoff_in_past_six_months',
                   'ji_live_in_domestic_service',
                   'job_info_foreign_ed','job_info_foreign_lang_req',
                   'job_info_experience']:
    
        new_data[column] = new_data[column].replace({'Y':True,'N':False})

    new_data.foreign_worker_info_education =             new_data['foreign_worker_info_education'].            replace({"None":np.nan,"-":np.nan,"--------":np.nan})
            
    new_data.foreign_worker_info_education_other =             new_data['foreign_worker_info_education_other'].                            replace({"None":np.nan,"-":np.nan,
                                            "--------":np.nan,
                                            "-----------":np.nan,
                                            "--------------":np.nan})
    
    #Change decision_date to just year, then convert back to datetime object
    new_data.decision_date = new_data.decision_date.dt.year
    new_data['decision_date'] = pd.to_datetime(new_data['decision_date'],
                                               format = "%Y")

    #Combine duplicate case_number data
    data_frame = pd.concat([
        new_data[new_data.duplicated('case_number')],
        new_data.loc[new_data.drop_duplicates('case_number',keep=False).index]
    ])
    data_frame = pd.concat([
        data_frame[data_frame.duplicated('case_number')],
        data_frame.loc[data_frame.drop_duplicates('case_number',
                                                  keep=False).index]
        ])
    
    data_frame = data_frame.set_index(data_frame.case_number)
    data_frame = data_frame.drop('case_number',axis = 1)
    
    return data_frame


# In[7]:

def feature_engineering(data_frame):
    '''Takes a data frame and removes rows of applicants that have "Withdrawn" as case
    status, replaces "Certified-Expired" with "Certified", changes "Other" to "Medical
    Degree" if applicable, and creates a column indicating if an agency was used.
    Parameters:
        data_frame (pd.Dataframe): the dataset being edited.
    Returns:
        A pd.DataFrame object with the above elements changed. 
    '''
    # Drop rows that have withdrawn -- don't need them for our analysis
    data_frame = data_frame.drop(data_frame[data_frame.case_status ==                                             'Withdrawn'].index)
    
    # Replaced "Certified-Expired" with "Certified"
    data_frame["case_status"] =                 data_frame["case_status"].replace('Certified-Expired', 'Certified')
    data_frame = data_frame.replace("", np.nan)
    
    # Find Medical Degree individuals. 
    expr = re.compile(r".*M.D.*|.*MED.*|.*MD.*")
    data_frame["foreign_worker_info_education_other"] =         data_frame["foreign_worker_info_education_other"].                                            replace(expr, "Medical Degree")
    
    # Add Medical Degree in education column.
    mask = data_frame["foreign_worker_info_education_other"] == "Medical Degree"
    data_frame.loc[mask,"foreign_worker_info_education"] =         data_frame[mask]["foreign_worker_info_education"].replace("Other", 
                                                                  "Medical Degree")
    data_frame = data_frame.drop("foreign_worker_info_education_other", 
                                                                    axis=1)
    
    # Determines if applicant used an agency
    data_frame["used_agency"] = ~data_frame["agent_firm_name"].isnull()
    data_frame = data_frame.drop(["agent_firm_name"], axis=1)
    
    return data_frame


# In[8]:

def count_data(column_name, data):
    '''Takes a specific variable from our data and 
        counts the amount of each unique category in the column. 
    Parameters:
        column_name (str): the name of the variable in the dataset.
        data (pd.Dataframe): the dataset being edited.
    Returns:
        A pd.DataFrame object with total count, certified count, and 
        certified rate. 
    '''
    certified = data[data["case_status"] == "Certified"]
    # Create a DataFrame with counts of each unique value of column.
    count = pd.DataFrame(pd.value_counts(data[column_name]))
    count.columns = ["Applied"]
    # Calculate the number certified applicants in each category. 
    cert = pd.DataFrame(certified.groupby(column_name).case_status.                                                            value_counts())
    cert = cert.rename(columns={"case_status": "certified_count"})
    count["Certified"] = cert.reset_index(level=[1])["certified_count"]
    # Calculate the ratio of certified to total applicant
    count["ratio"] = count["Certified"]/count["Applied"]
    return count


# In[9]:

def plot_summary_data(table, title, ylabel):
    '''Takes a table of data and plots a horizontal bar graph 
    of the number of applications.
    Parameters:
        table (pd.Dataframe): the dataframe used to plot.
        title (str): the title of graph.
        ylabel (str): the ylabel of graph (name of variable).
    '''
    # Create a bar graph.
    table[["Certified","Applied"]][::-1].plot(kind="barh",
                title=title, color=["indianred","lightseagreen"], 
                                              figsize=(10,5))
    plt.xlabel("Number of Visa Applications")
    plt.ylabel(ylabel)
    plt.tight_layout()
    # Show the bar graph. 
    plt.show()


# In[10]:

def plot_rate_data(table, title, ylabel):
    ''' Takes a table of data and plots a horizontal bar graph 
    of the rate of certification.
    Parameters:
        table (pd.Dataframe): the dataframe used to plot.
        title (str): the title of graph.
        ylabel (str): the ylabel of graph (name of variable).
    '''
    # Create a bar graph.
    table[["ratio"]].plot(kind="barh", title=title, color=["brown"], 
                          figsize=(10,5),legend=False)
    plt.xlabel("Certified Ratio")
    plt.ylabel(ylabel)
    plt.tight_layout()
    # Show the bar graph.
    plt.show()


# In[ ]:



