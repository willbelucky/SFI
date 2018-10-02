# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 10. 2.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ksif import Portfolio
from ksif.core.columns import *

TRADING_CAPITAL = 'trading_capital'

MIN_MAX_GP_A = 'min_max_gp_a'

MIN_RANK = 1
MAX_RANK = 15


def min_max_scale(series):
    min_value = np.min(series)
    max_value = np.max(series)
    normalized = (series - min_value) / (max_value - min_value)

    return normalized


SCORE = 'score'

pf = Portfolio()

# universe
universe = pf.loc[
           (pf[MKTCAP] < 800000000000) &
           (pf[MKTCAP] > 50000000000) &
           (pf[DATE] >= '2011-05-31') &
           (pf[ENDP] > 1000), :]
period_len = len(universe[DATE].unique())
universe[MIN_MAX_GP_A] = min_max_scale(universe[GP_A])
universe[TRADING_CAPITAL] = universe[TRADING_VOLUME_RATIO] * universe[MKTCAP]
kosdaq = universe.loc[universe[EXCHANGE] == '코스닥', :]
kosdaq.periodic_percentage(min_percentage=0.2, max_percentage=1.0, factor=FOREIGN_OWNERSHIP_RATIO)
kospi = universe.loc[universe[EXCHANGE] == '유가증권시장', :]

# Calculate score
## KOSDAQ
debt_ratio = ((kosdaq[DEBT_RATIO] > 0) & (kosdaq[DEBT_RATIO] < 1.5)) * 1
liq_ratio = (kosdaq[LIQ_RATIO] > 1) * 1
pcr = ((kosdaq[PCR] > 0) & (kosdaq[PCR] < 3)) * 1
pbr = ((kosdaq[PBR] > 0.5) & (kosdaq[PBR] < 1.5)) * 1
pgpr = ((kosdaq[PGPR] > 0) & (kosdaq[PGPR] < 5)) * 1
popr = ((kosdaq[POPR] > 0) & (kosdaq[POPR] < 15)) * 1
qroe = (kosdaq[QROE] > 0.03) * 1
roic = (kosdaq[ROIC] > 0.015) * 1
roaqoq = ((kosdaq[ROAQOQ] > 0.2) & (kosdaq[ROAQOQ] < 0.5)) * 1
gpa = (kosdaq[GP_A] > 0.3) * 1
mom1 = (kosdaq[MOM1] > 0) * 1
mom3 = (kosdaq[MOM3] > 0) * 1
mom6 = (kosdaq[MOM6_1] > 0) * 1
mom_average = (mom1 + mom3 + mom6)

kosdaq[SCORE] = debt_ratio + liq_ratio + pcr + 2 * pbr + pgpr + 3 * popr + qroe + roic + 2 * gpa + mom_average \
                + kosdaq[MIN_MAX_GP_A]

## KOSPI
debt_ratio = ((kospi[DEBT_RATIO] > 0) & (kospi[DEBT_RATIO] < 1.5)) * 1
liq_ratio = (kospi[LIQ_RATIO] > 1.5) * 1
pcr = ((kospi[PCR] > 0) & (kospi[PCR] < 3)) * 1
pbr = ((kospi[PBR] > 0) & (kospi[PBR] < 1.2)) * 1
pgpr = ((kospi[PGPR] > 0) & (kospi[PGPR] < 5)) * 1
popr = ((kospi[POPR] > 0) & (kospi[POPR] < 15)) * 1
qroe = (kospi[QROE] > 0.03) * 1
roic = (kospi[ROIC] > 0.02) * 1
roaqoq = ((kospi[ROAQOQ] > 0) & (kospi[ROAQOQ] < 0.6)) * 1
roayoy = (kospi[ROAYOY] > 0) * 1
gpa = (kospi[GP_A] > 0.3) * 1
mom1 = (kospi[MOM1] > 0) * 1
mom3 = (kospi[MOM3] > 0) * 1
mom6 = ((kospi[MOM6_1] > 0) & (kospi[MOM6_1] < 0.36)) * 1
mom_average = (mom1 + mom3 + mom6)
foreign_ownership_ratio = (kospi[FOREIGN_OWNERSHIP_RATIO] < 0.02) * 1

kospi[SCORE] \
    = debt_ratio + 2 * liq_ratio + pcr + 2 * pbr + pgpr + popr + qroe + roic + 2 * roayoy + 2 * gpa + 2 * foreign_ownership_ratio \
      + kospi[MIN_MAX_GP_A]


def calculate_rank_ic(portfolio: Portfolio, rolling: int = 6) -> pd.DataFrame:
    portfolio = portfolio.periodic_rank(min_rank=1, max_rank=10000, factor=SCORE, drop_rank=False)
    score_rank = "score_rank"
    portfolio = portfolio.rename(index=str, columns={"rank": score_rank})
    portfolio = portfolio.periodic_rank(min_rank=1, max_rank=10000, factor=RET_1, drop_rank=False)
    ret_1_rank = "ret_1_rank"
    portfolio = portfolio.rename(index=str, columns={"rank": ret_1_rank})
    rank_ic = portfolio.groupby(by=[DATE]).apply(
        lambda x: 1 - (6 * ((x[score_rank] - x[ret_1_rank]) ** 2).sum()) / (len(x) * (len(x) ** 2 - 1)))

    rank_ic = pd.DataFrame(rank_ic, columns=['rank_ic'])
    rank_ic['rolling_{}'.format(rolling)] = rank_ic['rank_ic'].rolling(window=rolling).mean()

    rank_ic.plot()
    plt.title('KOSPI Rank IC')
    plt.axhline(y=0, color='black')
    plt.show()

    return rank_ic


# periodic_rank(factor='score', 15)
kospi_rank_ic = calculate_rank_ic(kospi, 12)
kosdaq_rank_ic = calculate_rank_ic(kosdaq, 12)

kospi_rank_ic.plot()
plt.title('KOSPI Rank IC')
plt.axhline(y=0, color='black')
plt.show()

kosdaq_rank_ic.plot()
plt.title('KOSDAQ Rank IC')
plt.axhline(y=0, color='black')
plt.show()

# %% KOSPI
print("KOSPI")
kospi_outcome = kospi.outcome()
kospi.show_plot(title='Adapted KOSPI')
print(kospi_outcome)
print("CAGR:{}".format((kospi_outcome['total_return'] + 1) ** (12 / period_len) - 1))
kospi_simulation = kospi.quantile_distribution_ratio(factor=SCORE, cumulative=False, show_plot=False)
# print(kospi_selected.loc[kospi_selected[DATE] == '2018-08-31', [CODE, NAME, SCORE, ENDP, TRADING_CAPITAL]])
kospi.quantile_distribution_ratio(factor=SCORE, cumulative=True, show_plot=True, title='KOSPI 10 Quantile')

# %% KOSDAQ
print("KOSDAQ")
kosdaq_outcome = kosdaq.outcome()
kosdaq.show_plot(title='Adapted KOSDAQ')
print(kosdaq_outcome)
print("CAGR:{}".format((kosdaq_outcome['total_return'] + 1) ** (12 / period_len) - 1))
kosdaq_simulation = kosdaq.quantile_distribution_ratio(factor=SCORE, cumulative=False, show_plot=False)
# print(kosdaq_selected.loc[kosdaq_selected[DATE] == '2018-08-31', [CODE, NAME, SCORE, ENDP, TRADING_CAPITAL]])
kosdaq.quantile_distribution_ratio(factor=SCORE, cumulative=True, show_plot=True, title='KOSDAQ 10 Quantile')
