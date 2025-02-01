import jax
import jax.numpy as jnp

def f(x, y):
    return jnp.array([x**2 + y, x * y, x**3 + y**2])

# Forward-mode Jacobian
jacobian_fwd = jax.jacfwd(f)
jacobian_fwd_at = jacobian_fwd(2.0, 3.0)
print("Forward-mode Jacobian at (x=2, y=3):")
print(jacobian_fwd_at)

# Reverse-mode Jacobian
jacobian_rev = jax.jacrev(f)
jacobian_rev_at = jacobian_rev(2.0, 3.0)
print("Reverse-mode Jacobian at (x=2, y=3):")
print(jacobian_rev_at)
