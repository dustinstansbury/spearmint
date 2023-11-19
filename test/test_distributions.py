import numpy as np
from scipy import stats

from spearmint import distributions as dst

TEST_LABEL = "my_pdf"
TEST_COLOR = "blue"
TEST_VALUES = np.array([1, 1])


class MockDist:
    def rvs(self, size):
        return np.ones(size)

    def pdf(self, values):
        return np.ones(len(values))

    def pmf(self, values):
        return np.ones(len(values))

    def ppf(self, values):
        return np.ones(len(values))

    def cdf(self, values):
        return np.ones(len(values))


class ExtendedProbabilityDistribution(dst.ProbabilityDistribution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # This distribution will alwiays return 0. rvs
        self.dist = MockDist()

    def _get_values_grid(self):
        return TEST_VALUES

    def density(self, values):
        return TEST_VALUES


class ExtendedPdf(ExtendedProbabilityDistribution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ExtendedPmf(ExtendedProbabilityDistribution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def test_probability_distribution():
    pdf = ExtendedProbabilityDistribution(label=TEST_LABEL, color=TEST_COLOR)
    assert pdf.label == TEST_LABEL
    assert pdf.color == TEST_COLOR
    assert np.all(pdf.values_grid == TEST_VALUES)
    assert np.all(pdf.density(TEST_VALUES) == TEST_VALUES)

    vals, probs = pdf.get_series()
    assert np.all(vals == TEST_VALUES)
    assert np.all(probs == TEST_VALUES)

    assert np.all(pdf.sample(3) == np.array([1, 1, 1]))
    assert np.all(pdf.ppf(np.array([0])) == np.array([1]))


def test_pdf():
    pdf = ExtendedPdf(label=TEST_LABEL, color=TEST_COLOR)
    assert pdf.label == TEST_LABEL
    assert pdf.color == TEST_COLOR

    assert np.all(pdf.values_grid == TEST_VALUES)
    assert np.all(pdf.density(TEST_VALUES) == TEST_VALUES)

    vals, probs = pdf.get_series()
    assert np.all(vals == TEST_VALUES)
    assert np.all(probs == TEST_VALUES)

    assert np.all(pdf.sample(3) == np.array([1, 1, 1]))
    assert np.all(pdf.ppf(np.array([0])) == np.array([1]))

    # Pdf uses dist.pdf
    assert np.all(pdf.density(TEST_VALUES) == MockDist().pdf(TEST_VALUES))


def test_pmf():
    pmf = ExtendedPmf(label=TEST_LABEL, color=TEST_COLOR)
    assert pmf.label == TEST_LABEL
    assert pmf.color == TEST_COLOR

    assert np.all(pmf.values_grid == TEST_VALUES)
    assert np.all(pmf.density(TEST_VALUES) == TEST_VALUES)

    vals, probs = pmf.get_series()
    assert np.all(vals == TEST_VALUES)
    assert np.all(probs == TEST_VALUES)

    assert np.all(pmf.sample(3) == np.array([1, 1, 1]))
    assert np.all(pmf.ppf(np.array([0])) == np.array([1]))

    # Pmf uses dist.pmf
    assert np.all(pmf.density(TEST_VALUES) == MockDist().pmf(TEST_VALUES))


def test_probability_distribution_group():
    pmf = ExtendedPmf(label=TEST_LABEL, color=TEST_COLOR)
    pdf = ExtendedPdf(label=TEST_LABEL, color=TEST_COLOR)

    pdg = dst.ProbabilityDistributionGroup(pdfs=[pmf, pdf])

    for pd in pdg.pdfs:
        assert isinstance(pd, dst.ProbabilityDistribution)


def test_kde():
    np.random.seed(123)
    samples = np.random.randn(10000)
    kde = dst.Kde(samples=samples, label=TEST_LABEL, color=TEST_COLOR)
    values_grid = kde.values_grid
    assert values_grid.min() == samples.min()
    assert values_grid.max() == samples.max()
    assert kde.density(-100) == 0
    assert kde.density(100) == 0

    kde_samples = kde.sample(3)
    assert isinstance(kde_samples, np.ndarray)
    assert len(kde_samples) == 3


def test_gaussian_pdf():
    MEAN = 1
    STD = 2
    pdf = dst.Gaussian(mean=MEAN, std=STD, label=TEST_LABEL, color=TEST_COLOR)

    assert pdf.mean == MEAN
    assert pdf.std == STD
    assert pdf.label == TEST_LABEL
    assert pdf.color == TEST_COLOR

    grid_vals = pdf._get_values_grid()
    norm = stats.norm(MEAN, STD)
    assert grid_vals[0] == norm.ppf(dst.PDF_ZERO)
    assert grid_vals[-1] == norm.ppf(1 - dst.PDF_ZERO)


def test_bernoulli_pmf():
    P = 0.5
    pdf = dst.Bernoulli(p=P, label=TEST_LABEL, color=TEST_COLOR)

    assert pdf.p == P
    assert pdf.label == TEST_LABEL
    assert pdf.color == TEST_COLOR

    grid_vals = pdf._get_values_grid()
    assert grid_vals[0] == 0
    assert grid_vals[-1] == 1


def test_binomial_pmf():
    P = 0.5
    N = 10
    pdf = dst.Binomial(p=P, n=N, label=TEST_LABEL, color=TEST_COLOR)

    assert pdf.p == P
    assert pdf.n == N
    assert pdf.label == TEST_LABEL
    assert pdf.color == TEST_COLOR

    grid_vals = pdf._get_values_grid()
    binom = stats.binom(N, P)
    assert grid_vals[0] == binom.ppf(dst.PDF_ZERO)
    assert grid_vals[-1] == binom.ppf(1 - dst.PDF_ZERO)


def test_poisson_pmf():
    MU = 1
    pdf = dst.Poisson(mu=MU, label=TEST_LABEL, color=TEST_COLOR)

    assert pdf.mu == MU
    assert pdf.label == TEST_LABEL
    assert pdf.color == TEST_COLOR

    grid_vals = pdf._get_values_grid()
    poisson = stats.poisson(MU)
    assert grid_vals[0] == poisson.ppf(dst.PDF_ZERO)
    assert grid_vals[-1] == poisson.ppf(1 - dst.PDF_ZERO)
