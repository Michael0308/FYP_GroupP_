import numpy as np
import datetime
import yfinance as yf
import pandas as pd
import statistics as stat
import math
import pymoo
import pymoode
from pymoo.util.remote import Remote
from pymoo.core.problem import Problem
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoode.algorithms import GDE3
from pymoode.survival import RankAndCrowding
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymoo.visualization.scatter import Scatter
from pymoo.problems import get_problem
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.soo.nonconvex.pso_ep import EPPSO
from pymoo.indicators.hv import HV

class PortfolioProblem(Problem):

        def __init__(self, mu, dd, esg, **kwargs):
            super().__init__(n_var=len(mu), n_obj=3, xl=0.0, xu=1.0, n_ieq_constr=1, **kwargs)
            self.mu = mu
            self.dd = dd
            self.esg = esg

        def _evaluate(self, X, out, *args, **kwargs):

            exp_return = X.dot(self.mu).reshape([-1, 1])
        #    exp_sigma = np.sqrt(np.sum(X * X.dot(self.sigma), axis=1, keepdims=True))
            exp_dd = np.sqrt(np.sum((X*self.dd)**2, axis=1, keepdims=True))
            exp_esg = np.sum(X*esg, axis=1, keepdims=True)

            out["F"] = np.column_stack([exp_dd, -exp_return,-exp_esg])
            out["G"] = np.sum(X, axis=1) - 1
        def pareto_front(self, num_points=100):
            # Generate a grid of portfolio weights
            weights = np.linspace(0, 1, num_points)
            fronts = []

            for w1 in weights:
                for w2 in weights:
                    w3 = 1 - w1 - w2  # Ensure weights sum to 1
                    if w3 < 0:
                        continue
                    X = np.array([w1, w2, w3])
                    exp_return = X.dot(self.mu)
                    exp_dd = np.sqrt(np.sum((X * self.dd) ** 2))
                    exp_esg = np.sum(X * self.esg)
                    fronts.append([exp_dd, -exp_return, -exp_esg])
            return np.array(fronts)
def calculate_spacing(solutions):
        # Sort solutions based on the first objective (can be adjusted for multi-objective)
        sorted_solutions = solutions[np.argsort(solutions[:, 0])]

        # Calculate distances between consecutive solutions
        distances = np.linalg.norm(np.diff(sorted_solutions, axis=0), axis=1)

        # Calculate the average distance
        average_distance = np.mean(distances)

        # Calculate spacing
        spacing = np.sqrt(np.mean((distances - average_distance)**2))

        return spacing 

def get_esg_scores(ticker):
        try:
            # Validate the ticker symbol
            stock = yf.Ticker(ticker)
            # Attempt to get ESG scores
            esg_data = stock.sustainability.loc['totalEsg']
            if esg_data is not None and not esg_data.empty:
                return esg_data.iloc[0]
            else:
                return 0  # No ESG data found
        except Exception as e:
            return 0

def plotgraph(data):
    # Extracting x, y, z coordinates
    x = data[:, 0]
    y = -1*data[:, 1]
    z = -1*data[:, 2]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(x, y, z, c='b', marker='o')

    # Labels and title
    ax.set_xlabel('Downside Deviation')
    ax.set_ylabel('Expected Return')
    ax.set_zlabel('ESG Score')
    ax.set_title('SPEA2')
    plt.savefig("SPEA2.png",dpi=300)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    start_date = datetime.datetime(2023,3,1)
    end_date = datetime.datetime(2024,9,30)
    stocks = ['0005.HK','1299.HK','0939.HK','1398.HK','0388.HK','3988.HK','2318.HK','3968.HK','2388.HK','2628.HK',
        '0011.HK','0700.HK','9988.HK','3690.HK','1810.HK','9618.HK','9999.HK','0992.HK','9888.HK','0981.HK',
        '0285.HK','1211.HK','0669.HK','2015.HK','2020.HK','9633.HK','0027.HK','6690.HK','9961.HK','1876.HK',
        '2313.HK','0175.HK','0291.HK','1928.HK','0066.HK','2319.HK','2331.HK','0288.HK','6862.HK','1929.HK',
        '0322.HK','1044.HK','0881.HK','0016.HK','1109.HK','0823.HK','1113.HK','0688.HK','1997.HK','0012.HK',
        '0960.HK','1209.HK','0101.HK','0017.HK','0941.HK','0002.HK','0003.HK','0006.HK','2688.HK','0836.HK',
        '0762.HK','1038.HK','1093.HK','2269.HK','1177.HK','6618.HK','1099.HK','3692.HK','0241.HK','2359.HK',
        '0883.HK','0857.HK','0386.HK']
    returns = [0 for y in range(len(stocks))]
    log_returns = [0 for y in range(len(stocks))]
    for i in range(0, len(stocks)):
        data = yf.download(stocks[i], start=start_date, end=end_date, interval='1d')
        # # stocks_data = pd.concat([stocks_data, data])
        ret = [0 for y in range(data.shape[0])]
        
        for j in range(1, data.shape[0]):
            r = (data.iloc[j].iloc[0] - data.iloc[j-1].iloc[0])/data.iloc[j-1].iloc[0]
            ret[j] = r
            
            if j == data.shape[0]-1:
                returns[i] = ret
    returnsDf = pd.DataFrame(np.transpose(returns), columns=stocks)
    Rmean = np.array([stat.mean(returns[i]) for i in range(0,len(stocks))])
    N = 243
    cov_matrix = np.cov(returns)
    print(cov_matrix)
    downside_dev=[sum([j for j in returns[i] if j<0])**2/len(returns[i]) for i in range(len(stocks))]
    print(downside_dev)
    new_dd=np.array(downside_dev)
    #esg=[yf.Ticker(stock).sustainability.loc['totalEsg'].iloc[0] for stock in stocks]
    #esg=yf.Ticker("1299.HK").sustainability.loc['totalEsg']
    
    esg=np.array(list(map(get_esg_scores,stocks)))
    print(esg)
    mu = Rmean
    sigma = cov_matrix

    
    problem = PortfolioProblem(mu, new_dd, esg)
    
    algorithm = SPEA2(pop_size=300)
    res = minimize(problem,
                algorithm,
                termination=('n_gen', 200),
                save_history=True)


    plotgraph(res.F)


    weightDf = pd.DataFrame((res.X), columns=stocks)
    weightDf.to_csv('spea2.csv')
   

    ref_point = [0.01,0.002,50]
    ind = HV(ref_point=ref_point)
    print("HV", ind(res.F))

    spacing_value = calculate_spacing(res.F)
    print(f"Spacing Metric: {spacing_value:.4f}")