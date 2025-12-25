# ======================================================================================================================================================

# Part 2 of the code for the article Manuel Verdú, Óscar Carchano & Jesús Ruiz (2025) Detecting, characterizing, and predicting arbitrage opportunities in international rights issues,
# Research in International Business and Finance, 74, 102719.

# ======================================================================================================================================================

# To execute this code, you must have the following files available in https://doi.org10.5281/zenodo.18054892.
#   Verdu_Carchano_Ruiz_2025_Data.csv

# Once the data is available in the same directory, you only need to execute the code to obtain the results shown in the article.

# The article can be found at: https://doi.org/10.1016/j.ribaf.2024.102719

# ======================================================================================================================================================

#################### LIBRARIES TO USE ####################

import os
import warnings                         # Allows to ignore certain warnings and get a cleaner output.

import numpy as np                      # Allows to work with Series.
import pandas as pd						# Allows to organize the Data.
import statsmodels.api as sm            # Allows to apply regression methods.
import statsmodels.formula.api as smf	# Allows to apply regression methods.

from scipy import stats                 # Allows to work with statistical tests and distributions
from scipy.linalg import toeplitz

warnings.filterwarnings('ignore')
os.chdir('/Users/manuelverduhenares/Library/CloudStorage/Dropbox/1_Investigación/10_BBDD/2025_Verdu_Carchano_Ruiz')

#################### START OF THE CODE ####################

DATA = pd.read_csv('Verdu_Carchano_Ruiz_2025_Data.csv', delimiter = ';') # Open the data file.

Regions = ['AFR', 'AME', 'ASI', 'EUR']

Countries = ['EGY', 'SAU', 'TUN', 'BRA', 'CAN', 'USA', 'AUS', 'HKG', 'IND', 'MYS', 'NZL', 'PAK', 'SGP', 'LKA', 'AUT', 'BEL', 'DNK', 'FIN', 'FRA', 'DEU', 'GRC', 'ITA', 'NOR', 'POL', 'ESP', 'SWE', 'GBR']

##### MODELLING THE RETURNS FROM THE STRATEGY #####

### Defining the models to be estimated ###

Formula = 'ARR ~ DIL + IDX + ISC + CAP'     # General Model

Formula_DIL = 'ARR ~ IDX + ISC + CAP'       # Restricted model for DIL.
Formula_IDX = 'ARR ~ DIL + ISC + CAP'       # Restricted model for IDX.

Orders = [(1, 0, 0), (2, 1, 0), (1, 0, 0), (1, 0, 1)]   # Optimal ARIMA orders for each region.

### Estimating the models by regions ###

for R in range(0, len(Regions)):

    data = DATA[DATA['Region'] == Regions[R]]
    data = data.dropna(subset=['ARR', 'DIL', 'IDX', 'ISC', 'CAP'])
    data = data.reset_index(drop=True)

    ## Filtering the errors ##

    data = data[data['OUT'] == 0]
    data = data[data['DIL'] != 0]
    data = data[data['ISC'] != 0]
    data = data[data['CAP'] != 0]

    print(f'==================== OLS Estimation for: {Regions[R]} ====================')

    mod_ols = smf.ols(Formula, data)
    res_ols = mod_ols.fit(cov_type = 'HC1')
    ols_resid = res_ols.resid
    print(res_ols.summary())

    mod_ols_DIL = smf.ols(Formula_DIL, data)
    res_ols_DIL = mod_ols_DIL.fit(cov_type = 'HC1')
    DIL_ols_resid = res_ols_DIL.resid

    ols_KW_res_DIL = stats.kruskal(ols_resid, DIL_ols_resid)
    print(f'KW Test for DIL: {ols_KW_res_DIL[1]:.4f}')

    mod_ols_IDX = smf.ols(Formula_IDX, data)
    res_ols_IDX = mod_ols_IDX.fit(cov_type = 'HC1')
    IDX_ols_resid = res_ols_IDX.resid

    ols_KW_res_IDX = stats.kruskal(ols_resid, IDX_ols_resid)
    print(f'KW Test for IDX: {ols_KW_res_IDX[1]:.4f}')
    print()

    print(f'==================== FGLS Estimation for: {Regions[R]} ====================')

    mod_arima = sm.tsa.arima.ARIMA(endog = data['ARR'], order = Orders[R]) 
    res_arima = mod_arima.fit()

    rho = res_arima.params[1]
    
    order = toeplitz(range(len(data)))
    sigma = rho ** order

    mod_fgls = smf.gls(Formula, data, sigma = sigma)
    res_fgls = mod_fgls.fit(cov_type = 'HC1')
    fgls_resid = res_fgls.resid
    print(res_fgls.summary())

    mod_fgls_DIL = smf.gls(Formula_DIL, data, sigma = sigma)
    res_fgls_DIL = mod_fgls_DIL.fit(cov_type = 'HC1')
    DIL_fgls_resid = res_fgls_DIL.resid

    fgls_KW_res_DIL = stats.kruskal(fgls_resid, DIL_fgls_resid)
    print(f'KW Test for DIL: {fgls_KW_res_DIL[1]:.4f}')

    mod_fgls_IDX = smf.gls(Formula_IDX, data, sigma = sigma)
    res_fgls_IDX = mod_fgls_IDX.fit(cov_type = 'HC1')
    IDX_fgls_resid = res_fgls_IDX.resid

    fgls_KW_res_IDX = stats.kruskal(fgls_resid, IDX_fgls_resid)
    print(f'KW Test for IDX: {fgls_KW_res_IDX[1]:.4f}')
    print()

    print(f'==================== FOGLS Estimation for: {Regions[R]} ====================')

    Formula_Res = 'RES1 ~ RES0'
    Formula_T = 'ARRT ~ DILT + IDXT + ISCT + CAPT'

    Formula_DILT = 'ARRT ~ IDXT + ISCT + CAPT'
    Formula_IDXT = 'ARRT ~ DILT + ISCT + CAPT'

    N = len(data)

    arr_t = data['ARR']
    dil_t = data['DIL']
    idx_t = data['IDX']
    isc_t = data['ISC']
    cap_t = data['CAP']

    ARR_T = arr_t[1:] - rho * arr_t[:-1]
    DIL_T = arr_t[1:] - rho * dil_t[:-1]
    IDX_T = idx_t
    ISC_T = arr_t[1:] - rho * isc_t[:-1]
    CAP_T = arr_t[1:] - rho * cap_t[:-1]

    data_T = {'ARRT': ARR_T, 'DILT': DIL_T, 'IDXT': IDX_T, 'ISCT': ISC_T, 'CAPT': CAP_T}
    data_T = pd.DataFrame(data_T)

    mod_fogls = smf.ols(Formula_T, data_T)
    res_fogls = mod_fogls.fit(cov_type = 'HC1')
    fogls_resid = res_fogls.resid
    print(res_fogls.summary())

    mod_fogls_DIL = smf.ols(Formula_DILT, data_T)
    res_fogls_DIL = mod_fogls_DIL.fit(cov_type = 'HC1')
    DIL_fogls_resid = res_fogls_DIL.resid

    fogls_KW_res_DIL = stats.kruskal(fogls_resid, DIL_fogls_resid)
    print(f'KW Test for DIL: {fogls_KW_res_DIL[1]:.4f}')

    mod_fogls_IDX = smf.ols(Formula_IDXT, data_T)
    res_fogls_IDX = mod_fogls_IDX.fit(cov_type = 'HC1')
    IDX_fogls_resid = res_fogls_IDX.resid

    fogls_KW_res_IDX = stats.kruskal(fogls_resid, IDX_fogls_resid)
    print(f'KW Test for IDX: {fogls_KW_res_IDX[1]:.4f}')

#################### END OF THE CODE ####################