import numpy as np

# Settings for the stages (Leader acts first, then Followers sequence)
LEADER = 0
BUYER = 1
TRANSFORM = 2
stages = {LEADER, BUYER, TRANSFORM}

config = {
    # System parameters
    'num_agents': 3,
    'num_commodities': 12,
    
    # Physical/Chemical Parameters
    'alpha': 0.5,
    'beta': 1.5,
    'delta': 0.5,
    'LAMBDA': 0.5,
    'UC': 0.5,
    'TX_P': 0.5,
    'INIT_INV': 100,
    
    # Calibrated Reward Scaling (Prevents Gradient Underflow)
    'RWD_SCALE': 1e-6,             
    
    # BRL-3 / AHO Parameters (MuditGaur et al. 2025)
    'use_aho': True,               # Enables Approximate Hypergradient Optimization
    'tau': 10.0,                   # Hypergradient mixing coefficient (tau in BRL-3)
    'gamma_aho': 0.99,             # Discount factor used in the advantage estimate of (W_off - U_off)
    'lambda_penalty': 0.005,       # Calibrated penalty scale to prevent objective dominance
    'lr_leader': 1e-4,             # Slower leader learning rate (outer timescale)
    'lr_follower': 3e-4,           # Fast follower learning rate (inner timescale)
    'leader_update_frequency': 5,  # Timescale update ratio
    
    # Training parameters
    'gamma': 0.99,
    'num_steps': 1000,             # Steps per epoch
    'episode_length': 1000,
    'num_epochs': 100,
    'history_length': 5,
    'save_freq': 1,
    'seed': 2024,
    'price_factor': 1.,
}

def init_historical_data():
    historic_data = {}
    historic_data['spot_price'] = np.array([
        [config['price_factor']*0.5], [0.8], [1.], [3.], [20.], [4.], 
        [8.], [100.], [0.2], [1.2], [0.15], [1.173]
    ])
    return historic_data
