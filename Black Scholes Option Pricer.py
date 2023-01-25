import numpy as np
import math
import time
from pyfiglet import Figlet

class OptionPricing:

    def __init__(self, S0, E, T, rf, sigma, iterations):
        self.S0 = S0
        self. E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_option_simulation(self):
        # We will have 2 columns: First column has 0's and the second column will store the payoff
        # We will need the first column of 0's: Payoff function is max(0, S-E) for call option
        option_data = np.zeros([self.iterations, 2])

        # Dimensions: 1 dimensional array with as many items as the iterations
        rand = np.random.normal(0, 1, [1, self.iterations])

        # Equation for the S(t) stock price
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2) + self.sigma * np.sqrt(self.T) * rand)

        #We need S-E in order to calculate the max(S-E, 0)
        option_data[:,1] = stock_price - self.E

        # Average for the Monte-Carlo method, we can use np.amax() to return the max(0, S-E)
        average = np.sum(np.amax(option_data, axis = 1)) / float(self.iterations)

        # Need to use exp(-rT) discount factor
        return np.exp(-1.0 * self.rf * self.T) * average

    def put_option_simulation(self):

        # We will have 2 columns like the previous function
        option_data = np.zeros([self.iterations, 2])

        # Dimensions: 1 dimensional array with as many items as the iterations
        rand = np.random.normal(0, 1, [1, self.iterations])

        # Equation for the S(t) stock price
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2) + self.sigma * np.sqrt(self.T) * rand)

        # We need E-S in order to calculate the max (E-S, 0)
        option_data[:,1] = self.E - stock_price

        # Average for the Monte-Carlo method, we can use np.amax() to return the max(0, E-S)
        average = np.sum(np.amax(option_data, axis = 1)) / float(self.iterations)

        # Need to use exp(-rT) discount factor
        return np.exp(-1.0 * self.rf * self.T) * average

if __name__ == "__main__":
        
    f = Figlet(font='slant')
    print(f.renderText('Black-Scholes Option Pricer'))
    print("---------------------------------------------")
        
    S0 = int(input("> Enter the underlying stock price: "))
    E = int(input("> Enter the Strike Price: "))
    T = int(input("> Enter the time to Expiration: "))
    rf = float(input("> Enter the risk-free interest rate: "))
    sigma = float(input("> Enter the volatility of the underlying stock: "))
    iterations = int(input("> Enter the number of iterations in the Monte-Carlo Simulation: "))

    model = OptionPricing(S0, E, T, rf, sigma, iterations)
    print("Call option price with Monte-Carlo approach: ", model.call_option_simulation())
    print("Put option price with Monte-Carlo approach: ", model.put_option_simulation())
    