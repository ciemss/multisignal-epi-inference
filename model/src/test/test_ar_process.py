import torch
import pyro
from pyrenew.process import ARProcess

def test_ar_can_be_sampled():
    pyro.set_rng_seed(62)
    ar1 = ARProcess(5, torch.tensor([0.95]), torch.tensor([0.5]))
    ar2 = ARProcess(5, torch.tensor([0.05, 0.025, 0.025]), torch.tensor([0.5]))
    
    # Test sampling
    sample1 = ar1.sample(10, torch.tensor([50.0]))
    sample2 = ar2.sample(15, torch.tensor([50.0, 49.9, 48.2]))
    assert sample1.shape[0] == 10
    assert sample2.shape[0] == 15

def test_ar_samples_correctly_distributed():
    pyro.set_rng_seed(62)
    ar_mean = 5
    noise_sd = torch.tensor([0.5])
    ar_inits = torch.tensor([25.0])
    ar1 = ARProcess(ar_mean, torch.tensor([0.75]), noise_sd)
    long_ts = ar1.sample(50000, ar_inits)
   
    # Test if the first sample matches the provided initial value
    assert torch.isclose(long_ts[0], torch.tensor(25.0), atol=1e-5), f"Expected {long_ts[0]} to be close to 25.0"
    assert torch.abs(long_ts[-1] - ar_mean) < 4 * noise_sd, f"Expected final value to be close to the mean {ar_mean}, got {long_ts[-1]}"