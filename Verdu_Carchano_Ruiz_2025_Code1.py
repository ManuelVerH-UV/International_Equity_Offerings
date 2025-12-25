# ======================================================================================================================================================

# Part 1 of the code for the article Manuel Verdú, Óscar Carchano & Jesús Ruiz (2025) Detecting, characterizing, and predicting arbitrage opportunities in international rights issues,
# Research in International Business and Finance, 74, 102719.

# ======================================================================================================================================================

# To execute this code, you must have the following files available in https://doi.org10.5281/zenodo.18054892.
#   Verdu_Carchano_Ruiz_2025_Data.csv

# Once the data is available in the same directory, you only need to execute the code to obtain the results shown in the article.

# The article can be found at: https://doi.org/10.1016/j.ribaf.2024.102719

# ======================================================================================================================================================

#################### LIBRARIES TO USE ####################

import warnings                         # Allows to ignore certain warnings and get a cleaner output.

import numpy as np                      # Allows to work with Series.
import pandas as pd						# Allows to organize the Data.

from scipy import stats                 # Allows to work with statistical tests and distributions

warnings.filterwarnings('ignore')

#################### START OF COMPLEMENTARY FUNCTIONS ####################

def sample_description(DATA, S, Code):
    """
    Runs the description of the sample and the tests.
    """

    DATA = DATA[DATA['OUT'] == 0]
    DATA = DATA[DATA['DIL'] != 0]
    DATA = DATA[DATA['ISC'] != 0]
    DATA = DATA[DATA['CAP'] != 0]
    DATA = DATA.reset_index(drop = True)

    ### Full Sample ###
    if S == 'F':

        OUT = len(DATA[DATA['OUT'] == 1])   # Obtain the number of outliers.
        tst = calculate_stats(DATA)
  
        print(f'==================== {Code} - Full Sample ====================')
        print(f'N: {tst["N"]} / ARB: {tst["ARB"]} ({(tst["R"]*100):.2f}%)  - Outliers: {OUT}')
        print(f'Average: {tst["Mean"]*100:.2f}% ({tst["Std"]:.2f})% - t: {tst["T-test p-value"]:.4f} ({tst["Bootstrap CI Lower"]:.4f}, {tst["Bootstrap CI Upper"]:.4f})')
        print(f'Median: {tst["Median"]*100:.2f}% - W: {tst["Wilcoxon p-value"]:.4f}')
        print(f'Maximum: {tst["Maximum"]*100:.2f}% - Minimum: {tst["Minimum"]*100:.2f}% - Skewness: {tst["Skewness"]:.2f} - Kurtosis: {tst["Kurtosis"]:.2f}')
        print()

    ### Sample per Periods ###

    elif S == '1' or S == '2' or S == '3':

        tst = calculate_stats(DATA[DATA['PER'] == f'PER{S}'])
        print(f'==================== {Code} - PER{S} ====================')
        print(f'N: {tst["N"]} / ARB: {tst["ARB"]} ({(tst["R"]*100):.2f}%)')
        print(f'Average: {tst["Mean"]*100:.2f}% ({tst["Std"]:.2f})% - t: {tst["T-test p-value"]:.4f} ({tst["Bootstrap CI Lower"]:.4f}, {tst["Bootstrap CI Upper"]:.4f})')
        print(f'Median: {tst["Median"]*100:.2f}% - W: {tst["Wilcoxon p-value"]:.4f}')
        print()

    ### Dilutive Equity Offerings ###
    
    elif S  == 'DIL':

        tst = calculate_stats(DATA[DATA['DIL'] >= 0.5])
        print(f'==================== {Code} - DIL ====================')
        print(f'N: {tst["N"]} / ARB: {tst["ARB"]} ({(tst["R"]*100):.2f}%)')
        print(f'Average: {tst["Mean"]*100:.2f}% ({tst["Std"]:.2f})% - t: {tst["T-test p-value"]:.4f} ({tst["Bootstrap CI Lower"]:.4f}, {tst["Bootstrap CI Upper"]:.4f})')
        print(f'Median: {tst["Median"]*100:.2f}% - W: {tst["Wilcoxon p-value"]:.4f}')
        print()

    ### Non-Dilutive Equity Offerings ###
    
    elif S  == 'nDIL':

        tst = calculate_stats(DATA[DATA['DIL'] < 0.5])
        print(f'==================== {Code} - nDIL ====================')
        print(f'N: {tst["N"]} / ARB: {tst["ARB"]} ({(tst["R"]*100):.2f}%)')
        print(f'Average: {tst["Mean"]*100:.2f}% ({tst["Std"]:.2f})% - t: {tst["T-test p-value"]:.4f} ({tst["Bootstrap CI Lower"]:.4f}, {tst["Bootstrap CI Upper"]:.4f})')
        print(f'Median: {tst["Median"]*100:.2f}% - W: {tst["Wilcoxon p-value"]:.4f}')
        print()

    ### Companies listed in the main Stock Index of their country ###
    
    elif S  == 'IDX':

        tst = calculate_stats(DATA[DATA['IDX'] == 1])
        print(f'==================== {Code} - IDX ====================')
        print(f'N: {tst["N"]} / ARB: {tst["ARB"]} ({(tst["R"]*100):.2f}%)')
        print(f'Average: {tst["Mean"]*100:.2f}% ({tst["Std"]:.2f})% - t: {tst["T-test p-value"]:.4f} ({tst["Bootstrap CI Lower"]:.4f}, {tst["Bootstrap CI Upper"]:.4f})')
        print(f'Median: {tst["Median"]*100:.2f}% - W: {tst["Wilcoxon p-value"]:.4f}')
        print()

    ### Companies not listed in the main Stock Index of their country ###
    
    elif S  == 'nIDX':

        tst = calculate_stats(DATA[DATA['IDX'] == 0])
        print(f'==================== {Code} - nIDX ====================')
        print(f'N: {tst["N"]} / ARB: {tst["ARB"]} ({(tst["R"]*100):.2f}%)')
        print(f'Average: {tst["Mean"]*100:.2f}% ({tst["Std"]:.2f})% - t: {tst["T-test p-value"]:.4f} ({tst["Bootstrap CI Lower"]:.4f}, {tst["Bootstrap CI Upper"]:.4f})')
        print(f'Median: {tst["Median"]*100:.2f}% - W: {tst["Wilcoxon p-value"]:.4f}')
        print()

def calculate_stats(data):
    """
    Calculate descriptive statistics and hypothesis tests.
    """
    
    N = len(data)

    if N == 0:
        return {
            'N': 0,
            'ARB': np.nan,
            'R': np.nan,
            'Mean': np.nan,
            'Std': np.nan,
            'Median': np.nan,
            'Maximum': np.nan,
            'Minimum': np.nan,
            'Skewness': np.nan,
            'Kurtosis': np.nan,
            'T-test p-value': np.nan,
            'Wilcoxon p-value': np.nan,
            'Bootstrap CI Lower': np.nan,
            'Bootstrap CI Upper': np.nan
        }

    ARB = len(data[data['ARB'] == 1])   # Number of arbitrages.
    R = ARB / N if N > 0 else 0         # Proportion of arbitrages.

    values = data['ARR'].dropna()
    values = values.reset_index(drop=True)    

    # Main statistics.

    mean = values.mean()
    std = values.std()
    median = values.median()
    max_val = values.max()
    min_val = values.min()
    skewness = values.skew()
    kurtosis = values.kurtosis()
    
    # T-test (testing if mean is different from 0)
    t_stat, t_pvalue = stats.ttest_1samp(values, 0)
    
    # Wilcoxon signed-rank test (testing if median is different from 0)
    try:
        wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(values)

    except:
        wilcoxon_pvalue = np.nan

    # Bootstrap test

    n_bootstrap = 999
    bootstrap_T = []
    np.random.seed(42)
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(values, size=len(data), replace=True)
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_std = np.std(bootstrap_sample, ddof=1)
        bootstrap_t = bootstrap_mean / (bootstrap_std / np.sqrt(len(bootstrap_sample)))
        
        bootstrap_T.append(bootstrap_t)
    
    bootstrap_T = np.array(bootstrap_T)
    bootstrap_ci_lower = mean - np.percentile(bootstrap_T, 97.5) * (std / np.sqrt(len(values)))
    bootstrap_ci_upper = mean - np.percentile(bootstrap_T, 2.5) * (std / np.sqrt(len(values)))
    
    return {
        'N': N,
        'ARB': ARB,
        'R': R,
        'Mean': mean,
        'Std': std,
        'Median': median,
        'Maximum': max_val,
        'Minimum': min_val,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'T-test p-value': t_pvalue,
        'Wilcoxon p-value': wilcoxon_pvalue,
        'Bootstrap CI Lower': bootstrap_ci_lower,
        'Bootstrap CI Upper': bootstrap_ci_upper
    }

#################### END OF COMPLEMENTARY FUNCTIONS ####################

#################### START OF THE CODE ####################

DATA = pd.read_csv('Verdu_Carchano_Ruiz_2025_Data.csv', delimiter = ';') # Open the data file.

Regions = ['AFR', 'AME', 'ASI', 'EUR']

Countries = ['EGY', 'SAU', 'TUN', 'BRA', 'CAN', 'USA', 'AUS', 'HKG', 'IND', 'MYS', 'NZL', 'PAK', 'SGP', 'LKA', 'AUT', 'BEL', 'DNK', 'FIN', 'FRA', 'DEU', 'GRC', 'ITA', 'NOR', 'POL', 'ESP', 'SWE', 'GBR']

Samples = ['F', '1', '2', '3', 'DIL', 'nDIL', 'IDX', 'nIDX']

##### RESULTS FROM THE ARBITRAGE STRATEGY #####

### Sample Distribution by Arbitrage Results, Descriptive Statistics and Statistical Tests###

## Total Sample ##

Code = 'Total'

for Sample in Samples:
    sample_description(DATA, Sample, Code)

## Results for Regions ##

for Region in Regions:

    data = DATA[DATA['Region'] == Region]
    Code = Region

    for Sample in Samples:
        sample_description(data, Sample, Code)

for Country in Countries:

    data = DATA[DATA['Country'] == Country]
    Code = Country

    for Sample in Samples:
        sample_description(data, Sample, Code)

#################### END OF THE CODE ####################
