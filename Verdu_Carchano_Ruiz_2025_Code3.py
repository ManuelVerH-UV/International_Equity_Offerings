# ======================================================================================================================================================

# Part 3 of the code for the article Manuel Verdú, Óscar Carchano & Jesús Ruiz (2025) Detecting, characterizing, and predicting arbitrage opportunities in international rights issues,
# Research in International Business and Finance, 74, 102719.

# ======================================================================================================================================================

# To execute this code, you must have the following files available in https://doi.org10.5281/zenodo.18054892.
#   Verdu_Carchano_Ruiz_2025_Data.csv

# Once the data is available in the same directory, you only need to execute the code to obtain the results shown in the article.

# The article can be found at: https://doi.org/10.1016/j.ribaf.2024.102719

# ======================================================================================================================================================

#################### LIBRARIES TO USE ####################

import warnings                         # Allows to ignore certain warnings and get a cleaner output.
import itertools                        # Allows to work with combinations and permutations.

import pandas as pd						# Allows to organize the Data.
import statsmodels.api as sm            # Allows to apply regression methods.
import statsmodels.formula.api as smf	# Allows to apply regression methods.

from scipy.linalg import toeplitz

warnings.filterwarnings('ignore')

#################### START OF COMPLEMENTARY FUNCTIONS ####################

def success(FIT, DEP):
    """
    Calculate the success in predicting the result of an arbitrage opportunity.
    """

    Low = 1/3		# Maximum to consider a success in the estimation of a non-arbitrage.
    Up = 1 - Low	# Maximum to consider a success in the estimation of a arbitrage.

    # We obtain the total number of arbitrages and non-arbitrages.

    pos = 0
    neg = 0

    for a in range(0, len(DEP)):
        if DEP[a] == 1:
            pos = pos + 1
        elif DEP[a] == 0:
            neg = neg + 1

    F = len(FIT)
    PRD_P, PRD_N, PRD = [], [], []	# Number of success for each cathegory (arbitrage, non-arbitrage and total).

    for f in range(0,F):
        if (DEP[f] == 0 and FIT[f] <= Low):
            PRD_N.append(1)
            PRD.append(1)
        elif (DEP[f] == 1 and FIT[f] >= Up):
            PRD_P.append(1)
            PRD.append(1)

    # The success rate is obtained.
    
    suc_p = (sum(PRD_P) / pos)
    suc_n = (sum(PRD_N) / neg)
    suc = (sum(PRD) / len(DEP))

    return suc, suc_p, suc_n

def predict_mod(data, Model):
    """
    Obtaining the model that best precicts the number of opportunities of arbitrage.
    """

    ### Step 1: Main variables from the model. ###

    Variables1 = ('DIL', 'IDX', 'ISC', 'CAP', 'GEN', 'ACQ', 'INV', 'REF')
    Formulas1 = []
    SUC_P1 = []

    ## Obtaining all combinations of variables for the model. ##

    for i in range(1, len(Variables1) + 1):
        for combo in itertools.combinations(Variables1, i):
            formula = "ARB ~ " + " + ".join(combo)
            Formulas1.append(formula)

    ## Estimation of each model and obtention of the success rate. ##

    for Formula1 in Formulas1:

        try:

            if Model == 'LGT':
                mod = smf.logit(Formula1, data)
            elif Model == 'PRT':
                mod = smf.probit(Formula1, data)

            res = mod.fit(cov_type = 'HC1', disp = 0, warn_convergence = False) #white robustness.
            pred = res.predict()

            # We  extract the fitted values to annalyze the success of the model.

            FIT = res.fittedvalues.values
            DEP = data['ARB'].values.tolist()

            suc, suc_p, suc_n = success(FIT, DEP)

        except:

            suc_p = 0

        SUC_P1.append(suc_p)

    FormulaS = Formulas1[SUC_P1.index(max(SUC_P1))] + ' + '
    
    ### Step 2: Interations with the economic sector. ###

    ## Obtaining all combinations of variables for the new model. ##
    
    Variables2 = ('DIL * ACA', 'DIL * BAS', 'DIL * CYC', 'DIL * NCY', 'DIL * ENE', 'DIL * FIN', 'DIL * GOV', 'DIL * HEA', 'DIL * IND', 'DIL * EST', 'DIL * TEC', 'DIL * UTI')
    Formulas2 = []
    R2, SUC, SUC_P, SUC_N = [], [], [], []

    for i in range(1, len(Variables2) + 1):
        for combo in itertools.combinations(Variables2, i):
            formula = FormulaS + " + ".join(combo)
            Formulas2.append(formula)

    ## Estimation of each model and obtention of the success rate. ##

    for Formula2 in Formulas2:

        try:

            if Model == 'LGT':
                mod = smf.logit(Formula2, data)
            elif Model == 'PRT':
                mod = smf.probit(Formula2, data)

            res = mod.fit(cov_type = 'HC1', disp = 0, warn_convergence = False) #white robustness.
            pred = res.predict()
            r_squared = 1 - ((data['ARB'] - pred)**2).sum() / ((data['ARB'] - data['ARB'].mean())**2).sum()
            
            # We  extract the fitted values to annalyze the success of the model.

            FIT = res.fittedvalues.values
            DEP = data['ARB'].values.tolist()

            suc, suc_p, suc_n = success(FIT, DEP)

        except:

            r_squared = 0
            suc_p = 0
            suc_n = 0
            suc = 0

        R2.append(r_squared)
        SUC.append(suc)
        SUC_P.append(suc_p)
        SUC_N.append(suc_n)

    FormulaE = Formulas2[SUC_P.index(max(SUC_P))]
    R2S = R2[SUC_P.index(max(SUC_P))]
    SUC = SUC[SUC_P.index(max(SUC_P))]
    SUC_N = SUC_N[SUC_P.index(max(SUC_P))]
    SUC_P = SUC_P[SUC_P.index(max(SUC_P))]

    return FormulaE, R2S, SUC, SUC_P, SUC_N

#################### END OF COMPLEMENTARY FUNCTIONS ####################

#################### START OF THE CODE ####################

DATA = pd.read_csv('Verdu_Carchano_Ruiz_2025_Data.csv', delimiter = ';') # Open the data file.

Regions = ['AFR', 'AME', 'ASI', 'EUR']

Countries = ['EGY', 'SAU', 'TUN', 'BRA', 'CAN', 'USA', 'AUS', 'HKG', 'IND', 'MYS', 'NZL', 'PAK', 'SGP', 'LKA', 'AUT', 'BEL', 'DNK', 'FIN', 'FRA', 'DEU', 'GRC', 'ITA', 'NOR', 'POL', 'ESP', 'SWE', 'GBR']

Samples = ['F', '1', '2', '3']
Models = ['LGT', 'PRT']

##### SUCCESS OF THE MODELS #####

for C in range(0, len(Countries)):

    data0 = DATA[DATA['Country'] == Countries[C]]
    data0 = data0[data0['OUT'] == 0]
    data0 = data0[data0['DIL'] != 0]
    data0 = data0[data0['ISC'] != 0]
    data0 = data0[data0['CAP'] != 0]

    for Sample in Samples:

        if Sample == 'F':
            data = data0.copy()

        else:
            data = data0[data0['PER'] == f'PER{Sample}'].copy()

        if len(data) == 0:

            print(f'==================== Results for {Countries[C]} - Period: {Sample} ====================')
            print('No Results available.')
            print()

        else:

            for Model in Models:

                Formula, R, SUC, SUC_P, SUC_N = predict_mod(data, Model)
                
                if SUC == 0:
                    
                    print(f'==================== Results for {Countries[C]} - Model: {Model} - Period: {Sample} ====================')
                    print('No Results available.')
                    print()

                else:

                    print(f'==================== Results for {Countries[C]} - Model: {Model} - Period: {Sample} ====================')
                    print(f'Formula (R2: {R*100:.2f}%): {Formula}')
                    print(f'Success: ARB: {SUC_P*100:.2f}% - nARB: {SUC_N*100:.2f}% - Total: {SUC*100:.2f}%')
                    print()

#################### END OF THE CODE ####################
