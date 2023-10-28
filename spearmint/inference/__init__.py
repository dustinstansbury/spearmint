from spearmint.inference.inference_base import InferenceProcedure, InferenceResults


def get_inference_procedure(
    inference_method: str, **inference_params
) -> InferenceProcedure:
    _method = inference_method.lower().replace("-", "_").replace(" ", "_")
    if _method in ("means_delta"):
        from .frequentist.means_delta import MeansDelta as IP

    elif _method in ("proportions_delta"):
        from .frequentist.proportions_delta import ProportionsDelta as IP

    elif _method in ("rates_ratio"):
        from .frequentist.rates_ratio import RatesRatio as IP

    elif _method in ("bootstrap"):
        from .frequentist.bootstrap_delta import BootstrapDelta as IP

    elif _method in (
        "gaussian",
        "bernoulli",
        "binomial",
        "beta_binomial",
        "gamma_poisson",
        "student_t",
        "exp_student_t",
    ):
        from .bayesian.bayesian_delta import BayesianDelta as IP
    else:
        raise ValueError("Unknown inference method {!r}".format(inference_method))

    return IP(inference_method=inference_method, **inference_params)


__all__ = ["InferenceProcedure", "InferenceResults", "get_inference_procedure"]
