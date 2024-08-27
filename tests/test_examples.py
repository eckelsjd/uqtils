import matplotlib.pyplot as plt

from uqtils.example import normal_example, gradient_example, mcmc_example


def test_all_examples():
    normal_example()
    plt.close('all')

    gradient_example()
    mcmc_example()
    plt.close('all')
