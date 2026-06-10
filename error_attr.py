""" Script to attribute errors to individual variables (Fig. 4 of Qirui's paper from 2022).

FRdM, 2nd of July 2024 """


###################### IMPORT MODULES ######################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import functions

import importlib
importlib.reload(functions)


###################### DEFINE FUNCTIONS ######################

def error_attr(emi_model, tau_model, M_model, emi_c, tau_c, M_c, models_list):

    """ Calculates the error attribution from different processes.

    :param emi_model: dictionary with the emission output from models.
    :param tau_model: dictionary with the lifetime output from models.
    :param M_model: dictionary with the MEC/MAC output from models.
    :param emi_c: emission constrained value (float).
    :param tau_c: lifetime constrained value (float).
    :param M_c: MEC/MAC constrained value (float).
    :returns: dictionary with the error attribution for the different masks.
    """

    result = {}

    for mask in emi_c.keys():
        result[mask] = {}

        for model in models_list:
            result[mask][model] = {'e_emi': 0, 'e_lifetime': 0, 'e_M': 0, 'e_cross': 0, 'e_AOD': 0}

            error_var1 = emi_model[mask][model]-emi_c[mask]
            error_var2 = (tau_model[mask][model]*24*3600)-tau_c[mask]
            error_var3 = (M_model[mask][model]*1e3)-M_c[mask]

            result[mask][model]['e_emi'] = error_var1*tau_c[mask]*M_c[mask]
            result[mask][model]['e_lifetime'] = emi_c[mask]*error_var2*M_c[mask]
            result[mask][model]['e_M'] = emi_c[mask]*tau_c[mask]*error_var3
            result[mask][model]['e_cross'] = (error_var1*error_var2*error_var3 +
                                              emi_c[mask]*error_var2*error_var3 + error_var1*tau_c[mask]*error_var3 + error_var1*error_var2*M_c[mask])

            result[mask][model]['e_AOD'] = (result[mask][model]['e_emi'] + result[mask][model]['e_lifetime'] +
                                            result[mask][model]['e_M'] + result[mask][model]['e_cross'])

    return result


###################### CALCULATE ERRORS ######################

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

emissions = {'africa': functions.open_pickle_files(path_original, 'emi_total.pickle')['africa'],
             'amazon': functions.open_pickle_files(path_original, 'emi_total.pickle')['amazon']}
emi_BC_OA = {'africa': functions.open_pickle_files(path_regrid, 'emi_BC_OA.pickle')['africa'],
             'amazon': functions.open_pickle_files(path_regrid, 'emi_BC_OA.pickle')['amazon']}
lifetime = {'africa': functions.open_pickle_files(path_original, 'lifetime.pickle')['africa'],
             'amazon': functions.open_pickle_files(path_original, 'lifetime.pickle')['amazon']}
lifetime_BC_OA = {'africa': functions.open_pickle_files(path_regrid, 'lifetime_BC_OA.pickle')['africa'],
                  'amazon': functions.open_pickle_files(path_regrid, 'lifetime_BC_OA.pickle')['amazon']}
MEC = {'africa': functions.open_pickle_files(path_original, 'MEC.pickle')['africa'],
       'amazon': functions.open_pickle_files(path_original, 'MEC.pickle')['amazon']}
MAC = {'africa': functions.open_pickle_files(path_regrid, 'MAC.pickle')['africa'],
       'amazon': functions.open_pickle_files(path_regrid, 'MAC.pickle')['amazon']}

# constrained values
emissions_c = {'africa': 2.83693*1e-10, 'amazon': 2.11718*1e-10}                   # units: kg/m2 s
emissions_c_BC_OA = {'africa': 2.89543*1e-10, 'amazon': 1.82650*1e-10}             # units: kg/m2 s
lifetime_c = {'africa': 3.747*24*3600, 'amazon': 4.076*24*3600}                    # units: s
lifetime_c_BC_OA = {'africa': 3.205*24*3600, 'amazon': 3.375*24*3600}              # units: s
MEC_c = {'africa': 6.4142*1e3, 'amazon': 5.853*1e3}                                # units: m2/kg
MAC_c = {'africa': 0.8768*1e3, 'amazon': 0.5689*1e3}                               # units: m2/kg

error_AOD = error_attr(emissions, lifetime, MEC, emissions_c, lifetime_c, MEC_c, models_AOD)
error_AAOD = error_attr(emi_BC_OA, lifetime_BC_OA, MAC, emissions_c_BC_OA, lifetime_c_BC_OA, MAC_c, models_AAOD)



###################### MAKE FIGURE ######################

# convert dictionary to dataframe to make stacked bar plot
error_am = pd.DataFrame.from_dict(error_AAOD['amazon']).T
error_af = pd.DataFrame.from_dict(error_AAOD['africa']).T

error_af.iloc[:,:-1].plot(kind='bar', stacked=True, color=['mediumaquamarine', 'coral', 'lightsteelblue', 'hotpink'], width=0.7, legend=False)
for index, model in enumerate(error_AAOD['africa'].keys()):
    plt.scatter(index, error_AAOD['africa'][model]['e_AOD'], color='black')
plt.axhline(y=0, color='black', linewidth=0.4)
plt.ylim(-0.09, 0.09)
plt.xticks(np.arange(0, len(models_AAOD), 1), list(np.arange(1, len(models_AAOD)+1, 1)), rotation=0)
plt.ylabel('AAOD')
# plt.legend()
plt.title('AAOD error attribution Amazon')
plt.show(block=True)



###################### COMPARISON WITH QIRUI'S DATA ######################

# od550_am_f = np.zeros(len(models_list))
# od550_af_f = np.zeros(len(models_list))
# for i in range(len(models_list)):
#     od550_af_f[i] = error_af['e_AOD'][models_list[i]]
#     od550_am_f[i] = error_am['e_AOD'][models_list[i]]
#
# od550_am_q = np.array([-0.122893620749568, -0.014566142579173, 0.062606732825185, -0.085676331540202, -0.077504922171687, -0.072962511797999, -0.010644454976176, -0.211484570523356, -0.119537819644069, -0.107583720704173, -0.046436150094127, 0.035492937544728, 0.22006551502123, -0.148537163158511, 0.034122447947408, 0.15058092353716, -0.104491759558772])
# model_am = LinearRegression().fit(od550_am_f.reshape(-1, 1), od550_am_q)
# slope_am, intercept_am = model_am.coef_, model_am.intercept_
# r2_am = r2_score(od550_am_f, od550_am_q)
#
# od550_af_q = np.array([-0.279708137955689, -0.118005236830735, -0.084910383667969, -0.184663465466523, -0.286228796687149, -0.20085310026648, -0.223873069729828, -0.361263862099671, -0.302567085471177, -0.3269272653723, -0.272002091851258, -0.119968971457505, 0.067464837584472, -0.337185001220726, -0.226769587245011, 0.044840166125274, -0.275525501217865])
# model_af = LinearRegression().fit(od550_af_f.reshape(-1, 1), od550_af_q)
# slope_af, intercept_af = model_af.coef_, model_af.intercept_
# r2_af = r2_score(od550_af_f, od550_af_q)
#
# plt.figure()
# plt.scatter(od550_am_f, od550_am_q, color='blue')
# plt.plot(od550_am_f, slope_am*od550_am_f+intercept_am, color='blue', linewidth=0.7, label='amazon')
# plt.text(-0.08, -0.25, 'R2 score: '+str(r2_am), color='blue')
# plt.scatter(od550_af_f, od550_af_q, color='orange')
# plt.plot(od550_af_f, slope_af*od550_af_f+intercept_af, color='orange', linewidth=0.7, label='africa')
# plt.text(-0.08, -0.3, 'R2 score: '+str(r2_af), color='orange') #"%.2f" % a
# plt.axline((0, 0), slope=1, color='black')
# plt.xlabel('Fiona values')
# plt.ylabel('Qirui values')
# plt.legend()
# plt.title('AOD error regression')

abs550_am_f = np.zeros(len(models_list))
abs550_af_f = np.zeros(len(models_list))
for i in range(len(models_list)):
    abs550_af_f[i] = error_af['e_AOD'][models_list[i]]
    abs550_am_f[i] = error_am['e_AOD'][models_list[i]]

abs550_am_q = np.array([-0.0045832940426074, -0.0019877324961679, -0.0022415290123654, 0.0207292126023339, 0.0421861643855045, 0.0031500184440183, 0.056398838022131, 0.0191209121602465, 0.0208035619461106, -0.0063672304408323, 0.0098030935379153, 0.0078442472390775, -0.0092986559637063, -0.0098670547936436, -0.0084124919938926, 0.0070904368066358, 0.0100332410537766])
model_am = LinearRegression().fit(abs550_am_f.reshape(-1, 1), abs550_am_q)
slope_am, intercept_am = model_am.coef_, model_am.intercept_
r2_am = r2_score(abs550_am_f, abs550_am_q)

abs550_af_q = np.array([-0.0362986612071199, -0.0330988872357942, -0.0407060958123489, -0.0225453707999273, 0.0184094842248433, -0.0363387219286247, 0.0328486666447693, -0.02514895303972, -0.000161833656458388, -0.0465320762735569, -0.0300980741308557, -0.0309049170026073, -0.0516402758327152, -0.0506259054894948, -0.0498571899701466, -0.031415260923533, -0.0281502376920266])
model_af = LinearRegression().fit(abs550_af_f.reshape(-1, 1), abs550_af_q)
slope_af, intercept_af = model_af.coef_, model_af.intercept_
r2_af = r2_score(abs550_af_f, abs550_af_q)

plt.figure()
plt.scatter(abs550_am_f, abs550_am_q, color='blue')
plt.plot(abs550_am_f, slope_am*abs550_am_f+intercept_am, color='blue', linewidth=0.7, label='amazon')
plt.text(0.02, -0.037, 'R2 score: '+str(r2_am), color='blue')
plt.scatter(abs550_af_f, abs550_af_q, color='orange')
plt.plot(abs550_af_f, slope_af*abs550_af_f+intercept_af, color='orange', linewidth=0.7, label='africa')
plt.text(0.02, -0.042, 'R2 score: '+str(r2_af), color='orange') #"%.2f" % a
plt.axline((0, 0), slope=1, color='black')
plt.xlabel('Fiona values')
plt.ylabel('Qirui values')
plt.legend()
plt.title('AAOD error regression')



###################### MODEL - OBS ######################

diff_af = []
diff_am = []
for model in od550['africa'].keys():
    diff_af.append((abs550['africa'][model]-0.071)-error_AOD['africa'][model]['e_AOD'])
    diff_am.append((abs550['amazon'][model]-0.030)-error_AOD['amazon'][model]['e_AOD'])

plt.plot(diff_af, color='orange', label='africa', linewidth=0.7, linestyle='dashed')
plt.plot(diff_am, color='blue', label='amazon', linewidth=0.7, linestyle='dashed')
# plt.xlabel('Model')
# plt.ylabel('Difference')
# plt.title('Difference between the two ways')
plt.legend()




