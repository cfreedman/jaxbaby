"""
Solver for the analytic version of the neoclassical growth model with full depreciation (\delta = 1)

Solution is summarized by capital evolution k_{t + 1} = \alpha \beta y_{t} = \alpha \beta e^{z_{t}}k_{t}^{\alpha} l_{t}^{1 - \alpha}

c_{t} = y_{t} - k_{t + 1} = e^{z_{t}} k_{t}^{\alpha} l_{t}^{1 - \alpha} - k_{t + 1}
      = (1 - \alpha \beta) e^{z_{t}} k_{t}^{\alpha} l_{t}^{1 - \alpha}

Additive preferences over consumption and leisure u(c,l) = log(c) - \psi l^{\phi}

Labor optimality:

\psi \phi l^{\phi - 1} c = w = (1 - \alpha) k_{t}^{\alpha} l_{t}^{-\alpha} = (1 - \alpha) y_{t} / l_{t}

=> \psi \phi l_{t}^{\phi} (1 - \alpha \beta) y_{t} = (1 - \alpha) y_{t}
=> l_{t}^{\phi} = (1 - \alpha) / (\psi \phi (1 - \alpha \beta))

=> l_{t} = ((1 - \alpha) / (\psi \phi (1 - \alpha \beta)))^{1/\phi} = L

y_{t} = e^{z_{t}} k_{t}^{\alpha} l_{t}^{1 - \alpha}

      = e^{z_{t}} k_{t}^{\alpha} L^{1 - \alpha}

k_{t + 1} = \alpha \beta y_{t}
c_{t} = y_{t} - k_{t + 1}

This explains how to back out analytical solutio nfor a given period given state (z_{t}, k_{t})

"""
