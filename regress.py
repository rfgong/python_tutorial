# find_premia.py
# -------
# Runs regression and generates output CSVs
import pandas as pd
import statsmodels.formula.api as sm
import numpy as np
from collections import OrderedDict

convert_skill_zscore = False
# Setup databases to read in from
market = pd.read_csv("market_measures_tut.csv")
cog = pd.read_csv("skills_current_tut.csv")
# Merge cognism with market
cog = cog.merge(market, on=["DATE", "TICKER"])
# Add industry and monthly controls
dummy_ind = pd.get_dummies(cog['INDUSTRY'], prefix='ind')
ind_col = list(dummy_ind.columns.values)
cog = cog.join(dummy_ind.loc[:, ind_col[1]:])
dummy_month = pd.get_dummies(cog['DATE'], prefix='month')
month_col = list(dummy_month.columns.values)
cog = cog.join(dummy_month.loc[:, month_col[1]:])
# Covariates list
cov_list = list(cog.columns[54:])
skill_col = ["[0] Personal Coaching", "[1] Business Development", "[2] Logistics", "[3] Business Development", "[4] Digital Marketing", "[5] Administration", "[6] Hospitality", "[7] Business Development", "[8] Musical Production", "[9] Industrial Management", "[10] Human Resources (Junior)", "[11] Human Resources (Senior)", "[12] Visual Design", "[13] Data Analysis", "[14] Business Development", "[15] Recruiting", "[16] Education", "[17] Business Development", "[18] Operations Management", "[19] Middle Management", "[20] Pharmaceutical", "[21] Product Management", "[22] Healthcare", "[23] Sales", "[24] Insurance", "[25] Social Media and Communications", "[26] Web Development", "[27] Manufacturing and Process Management", "[28] Electrical Engineering", "[29] Legal", "[30] Graphic Design", "[31] Non-Profit and Community", "[32] Retail and Fashion", "[33] Real Estate", "[34] Military", "[35] Accounting and Auditing", "[36] Administration", "[37] IT Management and Support", "[38] Construction Management", "[39] Video and Film Production", "[40] CRM and Sales Management", "[41] Energy, Oil, and Gas", "[42] Mobile Telecommunications", "[43] Software Engineering", "[44] Banking and Finance", "[45] Web Design", "[46] Public Policy", "[47] Business Development", "[48] Technical Product Management", "[49] Sales Management"]

if convert_skill_zscore:
    for i in range(50):
        mean = np.mean(cog["S"+str(i)])
        std = np.std(cog["S"+str(i)])
        if std == 0:  # These skills result are not interpretable (Coefficients are always 0 anyways)
            continue
        cog["S"+str(i)] = (cog["S"+str(i)] - mean) / std

# Find AbnSkills
cols = {"DATE": [], "TICKER": []}
for i in range(50):
    cols["AS" + str(i)] = []
abn = pd.DataFrame(cols)
abn["DATE"] = cog["DATE"]
abn["TICKER"] = cog["TICKER"]
dependent_vars = ' + '.join(cov_list)
for i in range(50):
    reg = sm.ols(formula="S"+str(i)+" ~ "+dependent_vars, data=cog).fit()
    abn["AS" + str(i)] = reg.resid

# Extension Tobit for equal market and book liabilities
q_db = cog[['DATE', 'TICKER', 'TOB']]
out_cols = OrderedDict({"SKILLS": skill_col, "COEFFICIENT": [], "SE": [], "TSTAT": []})
m_q = abn.merge(q_db, on=["DATE", "TICKER"])
for i in range(50):
    reg = sm.ols(formula="TOB ~ "+"AS"+str(i), data=m_q).fit()
    out_cols["COEFFICIENT"].append(reg.params["AS"+str(i)])
    out_cols["SE"].append(reg.bse["AS"+str(i)])
    out_cols["TSTAT"].append(reg.tvalues["AS"+str(i)])
# Write output
out = pd.DataFrame(out_cols)
out.to_csv("tobit_ols.csv", index=False)


