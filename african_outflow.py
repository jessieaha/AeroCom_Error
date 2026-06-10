""" Script to study the outflow region.

FRdM, 13th of August 2024 """


###################### IMPORT MODULES ######################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import functions

import importlib
importlib.reload(functions)


###################### DEFINE FUNCTIONS ######################

def linear_f(x, y):

    """ Calculates a linear regression with the sklearn LinearRegression.fit().
        It can accept as many independent variables (x) as one desires.
        The fit function is: y = a.x1 + b.x2 + c.x3 + ... + d """

    model = LinearRegression().fit(x, y)
    return model.coef_, model.intercept_, model.score(x, y)

def linear_f1(X, a, b, c, d):

    """ Defines a linear regression with a fit function of:.
        y = a.x1.x2.x3 + b.x1.x2 + c.x3 + d """

    x_1, x_2, x_3 = X
    return a*x_1*x_2*x_3 + b*x_1*x_2 + c*x_3 + d

def find_regression(y, x1, x2, x3, constr1, constr2, constr3, regression='regression_1'):

    """ Finds the linear regression between y and 3 independent variables.
        Thought to be used for the AOD and AAOD in the outflow region.
        
    :param y: dictionary containing the independent variables.
    :param x1: dictionary containing the variable #1.
    :param x2: dictionary containing the variable #2.
    :param x3: dictionary containing the variable #3.
    :param constr1: constrained value for variable #1.
    :param constr2: constrained value for variable #2.
    :param constr3: constrained value for variable #3.
    :param regression: type of regression to use. It has to be 'LinearRegression' or 'curve_fit' (string).
    :returns: numerical value of y in the desired region.
    """

    assert regression in ['regression_1', 'regression_2'], f'var must be either regression_1 or regression_2 and not {regression}'

    y = np.array(list(y.values()))

    if regression == 'regression_1':
        x = np.array([list(x1.values()), list(x2.values()), list(x3.values())]).T
        coef, intercept, r2 = linear_f(x, y)
        y_o = coef[0]*constr1 + coef[1]*constr2 + coef[2]*constr3 + intercept

        return y_o, coef, intercept

    if regression == 'regression_2':
        X = np.vstack((np.array(list(x1.values())), np.array(list(x2.values())), np.array(list(x3.values()))))
        param, param_cov = curve_fit(linear_f1, X, y)
        y_o = param[0]*constr1*constr2*constr3 + param[1]*constr1*constr2 + param[2]*constr3 + param[3]

        return y_o, param


###################### FIND REGRESSIONS ######################

models_AOD = ['CAM5.3-Oslo_AP3-CTRL2016-PD', 'ECHAM6-HAM2_AP3-CTRL2016-PD', 'ECHAM6-SALSA_CTRL2016-PD',
              'ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD', 'TM5_AP3-CTRL2016',
              'CAM5-ATRAS_AP3-CTRL',  'EC-Earth3-AerChem-met2010_AP3-CTRL2019', 'ECHAM6.3-HAM2.3-met2010_AP3-CTRL',
              'ECHAM6.3-SALSA2.0-met2010_AP3-CTRL', 'GEOS-i33p2-met2010_AP3-CTRL', 'GFDL-AM4-met2010_AP3-CTRL',
              'GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010', 'GISS-ModelE2p1p1-OMA_AP3-CTRL-2010', 'INCA_AP3-CTRL',
              'NorESM2-met2010_AP3-CTRL', 'MIROC-SPRINTARS_AP3-CTRL', 'TM5-met2010_AP3-CTRL2019']
models_AAOD = ['CAM5.3-Oslo_AP3-CTRL2016-PD', 'CAM5_CTRL2016', 'CAM5-ATRAS_AP3-CTRL',
               'EC-Earth3-AerChem-met2010_AP3-CTRL2019', 'ECHAM6-HAM2_AP3-CTRL2016-PD',
               'ECHAM6.3-HAM2.3-met2010_AP3-CTRL', 'ECHAM6-SALSA_CTRL2016-PD',  'ECHAM6.3-SALSA2.0-met2010_AP3-CTRL',
               'ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD', 'GEOS-i33p2-met2010_AP3-CTRL', 'GFDL-AM4-met2010_AP3-CTRL',
               'GISS-ModelE2p1p1-OMA_AP3-CTRL-2010', 'INCA_AP3-CTRL', 'SPRINTARS-T213_AP3-CTRL2016-PD',
               'MIROC-SPRINTARS_AP3-CTRL', 'TM5_AP3-CTRL2016', 'TM5-met2010_AP3-CTRL2019']

path_original = '/Users/fionaromandemiguel/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/VU/data/var_files/original/0average/'
path_regrid = '/Users/fionaromandemiguel/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/VU/data/var_files/regrid/0average/'

AOD = functions.open_pickle_files(path_original, 'od550cs.pickle')['outflow_af']
AAOD = functions.open_pickle_files(path_regrid, 'abs550.pickle')['outflow_af']

emissions = functions.open_pickle_files(path_original, 'emi_total.pickle')['africa']
emi_BC_OA = functions.open_pickle_files(path_regrid, 'emi_BC_OA.pickle')['africa']
lifetime = functions.open_pickle_files(path_original, 'lifetime.pickle')['africa']
lifetime_BC_OA = functions.open_pickle_files(path_regrid, 'lifetime_BC_OA.pickle')['africa']
MEC = functions.open_pickle_files(path_original, 'MEC.pickle')['africa']
MAC = functions.open_pickle_files(path_regrid, 'MAC.pickle')['africa']

colors_AOD = functions.open_pickle_files('/Users/fionaromandemiguel/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/VU/data/var_files/',
                                         'colors_AOD.pickle')
colors_AAOD = functions.open_pickle_files('/Users/fionaromandemiguel/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/VU/data/var_files/',
                                          'colors_AAOD.pickle')

E_c, E_BC_OA_c = 2.85774e-10, 2.887036e-10
lifetime_c, lifetime_BC_OA_c = 3.7472, 3.2050
MEC_c, MAC_c = 6.3812, 0.8806
AOD_obs, AAOD_obs = 0.3993, 0.0642     # POLDER obs over outflow

# calculate mean of AeroCom models
mean = []
colors_list = []
for model in models_AOD:
    mean.append(AOD[model])
    colors_list.append(colors_AOD[model]['color'])

AOD_o = find_regression(AOD, emissions, lifetime, MEC, E_c, lifetime_c, MEC_c, 'regression_2')
AAOD_o = find_regression(AAOD, emi_BC_OA, lifetime_BC_OA, MAC, E_BC_OA_c, lifetime_BC_OA_c, MAC_c, 'regression_2')
# AOD_qirui = -0.05*E_c*lifetime_c*MEC_c + 0.16*E_c*lifetime_c + 0.0086*MEC_c + 0.36
# AAOD_qirui = -0.001249*E_BC_OA_c*lifetime_BC_OA_c*MAC_c - 2.6e-5*E_BC_OA_c*lifetime_BC_OA_c - 0.00538*MAC_c - 0.0021

plt.bar(np.arange(len(models_AOD))+1, np.array(list(AOD.values())), align='center', color=colors_list)
plt.ylim(0, 1)
plt.xlim(0.5, 17.5)
plt.xticks(np.arange(1, 18, 1))
plt.hlines(np.mean(mean), 0, 17, color='black', linestyles='dashed', label='Average default')
plt.hlines(AOD_obs, 0, 17, color='red', linestyles='dashed', label='POLDER obs')
plt.hlines(AOD_o[0], 0, 17, color='blue', linestyles='dashed', label='AOD corrected')
# plt.hlines(AOD_qirui, 0, 17, color='green', linestyles='dashed', label='AOD corrected Qirui')
plt.legend()
plt.text(12, 0.73, 'AOD obs: ' + str(AOD_obs), color='black')
plt.text(12, 0.67, 'AOD corrected: '+"%.3f" % AOD_o[0], color='black')
plt.ylabel('Africa outflow AOD')
plt.show(block=True)


###################### COMPARE WITH MODELLED AOD_o ######################

estimated_1 = []
for model in models_AOD:
    estimated_1.append(AOD_o[1][0]*emissions[model]*lifetime[model]*MEC[model] + AOD_o[1][1]*emissions[model]*lifetime[model] +
                       AOD_o[1][2]*MEC[model] + AOD_o[1][3])
x_model = np.array(list(AOD.values()))

slope_1, intercept_1, r2_1 = linear_f(x_model.reshape(-1, 1), estimated_1)

plt.scatter(AOD.values(), estimated_1, color=colors_list)
plt.axline((0, 0), slope=1, color='grey', linewidth=0.7, linestyle='dashed')
plt.plot(x_model, slope_1*x_model+intercept_1, color='black', linewidth=0.5)
plt.text(0.4, 0.02, 'Pearson coef: '+"%.4f" % pearsonr(x_model, estimated_1)[0], color='black')
# plt.legend()
plt.xlabel('Modelled AOD outflow')
plt.ylabel('Estimated AOD outflow')
plt.show(block=True)


###################### QIRUI'S DATA ######################

"""df = pd.read_excel('/Users/fionaromandemiguel/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/VU/new_data/aerocom_data.xlsx',
                   sheet_name='Qirui')
models_AOD_q = ['CAM5-ATRAS_AP3-CTRL', 'CAM5.3-Oslo_AP3-CTRL2016-PD', 'EC-Earth3-AerChem-met2010_AP3-CTRL2019',
                'ECHAM6-HAM2_AP3-CTRL2016-PD', 'ECHAM6.3-HAM2.3-met2010_AP3-CTRL', 'ECHAM6-SALSA_CTRL2016-PD',
                'ECHAM6.3-SALSA2.0-met2010_AP3-CTRL', 'ECMWF-IFS-CY42R1-CAMS-RA-CTRL_AP3-CTRL2016-PD',
                'GEOS-i33p2-met2010_AP3-CTRL', 'GFDL-AM4-met2010_AP3-CTRL', 'GISS-ModelE2p1p1-MATRIX_AP3-CTRL-2010',
                'GISS-ModelE2p1p1-OMA_AP3-CTRL-2010', 'INCA_AP3-CTRL', 'NorESM2-met2010_AP3-CTRL',
                'MIROC-SPRINTARS_AP3-CTRL', 'TM5_AP3-CTRL2016', 'TM5-met2010_AP3-CTRL2019']
emi_q = list(df['Unnamed: 3'][20:37])
tau_q = list(df['Unnamed: 4'][20:37])
MEC_q = list(df['Unnamed: 5'][20:37])

qirui = []
for model in range(len(models_AOD_q)):
    e = emi_q[model]*1e-11
    qirui.append(-0.05*e*tau_q[model]*MEC_q[model] + 0.16*e*tau_q[model] + 0.0086*MEC_q[model] + 0.36)
x_model = np.array(list(df['Unnamed: 6'][20:37]))

# slope_1, intercept_1, r2_1 = linear_f(x_model.reshape(-1, 1), estimated_1)
slope_q, intercept_q, r2_q = linear_f(x_model.reshape(-1, 1), qirui)

# plt.scatter(AOD.values(), estimated_1, color='blue')
plt.scatter(list(df['Unnamed: 6'][20:37]), qirui, color=colors_list)
plt.axline((0.15, 0.15), slope=1, color='grey', linewidth=0.7, linestyle='dashed')
# plt.plot(x_model, slope_1*x_model+intercept_1, color='black', linewidth=0.5, label='way 1')
plt.plot(x_model, slope_q*x_model+intercept_q, color='black', linewidth=0.7, label='way qirui')
plt.text(0.5, 0.2, 'Pearson coef: '+"%.4f" % pearsonr(x_model, qirui)[0], color='black')
plt.xlabel('Qirui Modelled AOD outflow')
plt.ylabel('Qirui Estimated AOD outflow')
plt.x"""