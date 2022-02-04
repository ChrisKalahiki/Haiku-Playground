import jax as jnp
from jax import jacfwd, jacrev
from jax.scipy.stats.multivariate_normal import logpdf as mvn_logpdf
import jaxopt

# log-likelihood for a multivariant normal
def log_liklihood_func(x):
    return mvn_logpdf(x,
                        jnp.array([1.0, 3.0]), #means
                        jnp.array([[2.0, 0.0], #covariance
                                [0.0, 1.0]]))

# get the maximum likelihood estimate
def MLE(ll_f, shape):
    solver = jaxopt.LBFGS(fun=lambda x: -ll_f(x))
    params, _ = solver.run(jnp.zeros(shape))
    return params

# Matrix of 2nd order partial derivatives
def hessian(f):
    return jacfwd(jacrev(f))

# Fisher information is the negative
# Hessian of the log-liklihood at the MLE
def fisher_info(ll_f, shape):
    mle_params = MLE(ll_f, shape)
    return -hessian(ll_f)(mle_params)

# We can approximate the standard deviation
def laplace_approx_sd(ll_f, shape):
    return jnp.sqrt(
        # 1/jnp.diag(fisher_info(ll_f, shape)) # this is the same as below
        jnp.diag(jnp.linalg(inv(fisher_info(ll_f, shape))))
    )

# square this to get back that original variance
laplace_approx_sd(log_liklihood_func, (2,))**2