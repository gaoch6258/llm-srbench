import numpy as np
import scipy.ndimage

def pde_update(g: np.ndarray, S: np.ndarray, dx: float, dy: float, dt: float, params: np.ndarray) -> np.ndarray:
    import numpy as np
    import scipy.ndimage

    # Extract parameters
    p0 = params[0]  # diffusion coefficient
    p1 = params[1]  # chemotactic sensitivity
    p2 = params[2]  # nonlinear decay term
    
    # Compute spatial derivatives
    dg_dx, dg_dy = np.gradient(g, dx, dy, axis=(0,1))
    laplace_g = scipy.ndimage.laplace(g, axes=(0,1)) / (dx**2)
    
    # Compute gradient of chemoattractant field
    dS_dx, dS_dy = np.gradient(S, dx, dy, axis=(0,1))
    
    # Compute gradient of g (for chemotaxis)
    grad_g = np.stack([dg_dx, dg_dy], axis=-1)
    grad_S = np.stack([dS_dx, dS_dy], axis=-1)
    
    # Compute chemotactic flux divergence: ∇·(g·∇S)
    # g·∇S = g * ∇S
    flux_x = g * dS_dx
    flux_y = g * dS_dy
    # Divergence: ∂(g·∂S/∂x)/∂x + ∂(g·∂S/∂y)/∂y
    div_flux = np.gradient(flux_x, dx, dy, axis=0)[0] + np.gradient(flux_y, dx, dy, axis=1)[0]
    
    # PDE: ∂g/∂t = D·Δg - χ·∇·(g·∇S) - r·g²
    dg_dt = p0 * laplace_g - p1 * div_flux - p2 * g**2
    
    # Forward Euler update
    g_next = g + dt * dg_dt
    
    return g_next


g = np.zeros((10, 10, 128))
S = np.zeros((10, 10, 128))
dx = 1.0
dy = 1.0
dt = 0.01
params = np.array([0.1, 0.1, 0.1, 0.1])
g_next = pde_update(g, S, dx, dy, dt, params)
print(g_next.shape)