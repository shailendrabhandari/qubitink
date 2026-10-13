# Mathematical model and visual interpretation

This document describes `quantum_wavepacket.py`, version 2, with the `filaments` style. The wavefunction is evaluated analytically. Only the streamline integration and rendering are numerical.

## 1. Hamiltonian and units

In dimensionless units with $`\hbar=m=1`$, the particle obeys

$$
i\partial_\tau\psi(\mathbf x,\tau)=\hat H\psi(\mathbf x,\tau),\qquad
\hat H=\frac12\sum_{i=1}^3\left(-\partial_{x_i}^2+\omega_i^2x_i^2\right),
$$

with

$$
(\omega_1,\omega_2,\omega_3)=(1,\sqrt2,\sqrt3).
$$

Here $`\tau`$ is model time. Physical units can be restored by choosing a reference angular frequency $`\omega_0`$, a mass $`m`$, the length scale $`\sqrt{\hbar/(m\omega_0)}`$, and the time scale $`1/\omega_0`$. No particular particle or experimental trap is specified.

## 2. Initial coherent states

Eight product coherent states are used. For $`k=0,\ldots,7`$, define

$$
\theta_k=\frac{2\pi k}{8},
$$

$$
\mathbf q_k(0)=\left(1.4\cos\theta_k,\;1.4\sin\theta_k,\;0.8\sin2\theta_k\right),
$$

$$
\mathbf p_k(0)=\left(-2.2\sin\theta_k,\;2.2\cos\theta_k,\;1.3\cos2\theta_k\right),
\qquad
\boldsymbol\alpha_k(0)=\frac{\mathbf q_k(0)+i\mathbf p_k(0)}{\sqrt2}.
$$

The vectors $`\mathbf q_k`$ and $`\mathbf p_k`$ are oscillator-scaled quadratures, not the unscaled position and momentum. The centre of each component has $`\langle x_i\rangle_k=q_{ki}/\sqrt{\omega_i}`$ and $`\langle\hat p_i\rangle_k=\sqrt{\omega_i}\,p_{ki}`$.

The initial complex weights are

$$
r_k=\exp\!\left[i\left(3\theta_k+0.55\sin3\theta_k\right)\right].
$$

Coherent states overlap, so normalizing the coefficients with $`\sum_k|r_k|^2`$ would be incorrect. The overlap matrix is

$$
G_{kl}=\langle\boldsymbol\alpha_k|\boldsymbol\alpha_l\rangle
=\exp\!\left[-\frac{\|\boldsymbol\alpha_k\|^2+\|\boldsymbol\alpha_l\|^2}{2}
+\overline{\boldsymbol\alpha_k}\cdot\boldsymbol\alpha_l\right].
$$

The code sets $`c_k=r_k/\sqrt{\mathbf r^\dagger G\mathbf r}`$, giving $`\mathbf c^\dagger G\mathbf c=1`$.

## 3. Exact evolution and position-space wavefunction

Each coherent amplitude rotates independently:

$$
\alpha_{ki}(\tau)=\alpha_{ki}(0)e^{-i\omega_i\tau},\qquad
q_{ki}(\tau)=\sqrt2\,\mathrm{Re}\alpha_{ki}(\tau),\qquad
p_{ki}(\tau)=\sqrt2\,\mathrm{Im}\alpha_{ki}(\tau).
$$

Equivalently, $`q_{ki}(\tau)=q_{ki}(0)\cos\omega_i\tau+p_{ki}(0)\sin\omega_i\tau`$ and $`p_{ki}(\tau)=p_{ki}(0)\cos\omega_i\tau-q_{ki}(0)\sin\omega_i\tau`$.

Writing $`\xi_i=\sqrt{\omega_i}x_i`$ and $`N=\prod_i(\omega_i/\pi)^{1/4}`$, the exact wavefunction is

$$
\psi(\mathbf x,\tau)=N e^{-\frac{i\tau}{2}\sum_i\omega_i}
\sum_{k=0}^7c_k
\exp\!\left[-\frac12\|\boldsymbol\xi-\mathbf q_k(\tau)\|^2
+i\left(\mathbf p_k(\tau)\cdot\boldsymbol\xi-\frac12\mathbf q_k(\tau)\cdot\mathbf p_k(\tau)\right)\right].
$$

This solves the stated Schrödinger equation without a time-stepping approximation. Since the coherent overlaps remain constant under these rotations, normalization is conserved. The evolution is confined and has no dissipative convergence to an attractor. The unequal, incommensurate frequencies produce quasiperiodic structure, not chaotic dynamics.

## 4. Density, interference, and current

The probability density is $`\rho=|\psi|^2`$. If $`\phi_k`$ denotes the normalized position-space coherent component, then

$$
\rho = \sum_{k} |c_k|^{2} |\phi_k|^{2}
+ 2\,\mathrm{Re}\left(
\sum_{k<l} \overline{c_k} c_l \overline{\phi_k} \phi_l
\right).
$$

The cross terms produce the interference pattern. Density has units of inverse volume when physical units are restored.

For this scalar Hamiltonian, the probability current is

$$
\mathbf j=\mathrm{Im}(\overline{\psi}\nabla\psi),\qquad
\partial_\tau\rho+\nabla\cdot\mathbf j=0.
$$

In physical units the current has an additional factor $`\hbar/m`$. For a component term $`u_k`$ in the sum above, its derivative is

$$
\partial_{x_i}u_k=\sqrt{\omega_i}\left[-(\xi_i-q_{ki})+ip_{ki}\right]u_k.
$$

The shaders evaluate these analytic derivatives, the density, the current, and $`\nabla\rho=2\mathrm{Re}(\overline{\psi}\nabla\psi)`$ directly. There is no stochastic dynamics or quantum measurement in this calculation.

## 5. What the curves mean

At each frame, model time is held fixed and a separate curve is integrated from each geometric seed:

$$
\frac{d\mathbf r}{ds}=\frac{\mathbf j(\mathbf r,\tau)}{\|\mathbf j(\mathbf r,\tau)\|}.
$$

Here $`s`$ is an arclength parameter. The implementation regularizes the denominator with $`\max(\|\mathbf j\|,10^{-12})`$ and uses midpoint RK2:

$$
\mathbf v(\mathbf r)=\frac{\mathbf j(\mathbf r,\tau)}{\max(\|\mathbf j(\mathbf r,\tau)\|,10^{-12})},\qquad
\mathbf r_{n+1}=\mathbf r_n+h\,\mathbf v\!\left(\mathbf r_n+\frac h2\mathbf v(\mathbf r_n)\right).
$$

The default has 256 samples per curve and $`h=6/(256-1)`$. There are 128 deterministic seeds, with eight heights between $`-1.3`$ and $`1.3`$ and 16 angular positions at each height. For angular index $`j`$, $`\beta=2\pi j/16+0.25z`$ and $`R=1.1+0.55\cos(1.1z)+0.12\sin(3\beta)`$, giving seed $`(R\cos\beta,R\sin\beta,z)`$. These seeds do not sample $`\rho`$.

A curve stops advancing when $`\rho<2\times10^{-5}`$ or $`\|\mathbf r\|>3.45`$. These cutoffs serve the visualization and do not bound the quantum state. At a zero of the current, a streamline direction is undefined.

The curves are instantaneous streamlines, recomputed at every frame. They are not particle trajectories. In particular, they are not obtained by integrating the time-dependent velocity $`\mathbf j/\rho`$. Animated brightness pulses indicate direction along the curve and do not represent physical speed or transported probability. Because current magnitude is removed from the integration direction, curve length does not measure current strength.

## 6. Surfaces, colour, and film time

The two displayed density surfaces are $`\rho=0.010`$ and $`\rho=0.035`$, in the model's dimensionless coordinates. A ray marcher detects crossings, then refines each detected crossing by seven bisection steps. This is a numerical surface rendering, so sufficiently fine features can be missed by the initial 128 ray samples.

Colour is a chosen cyan–violet–orange mapping of the phase of the summed wavefunction. The shader omits the common zero-point phase, so it displays

$$
\arg\!\left[e^{\frac{i\tau}{2}\sum_i\omega_i}\psi(\mathbf x,\tau)\right].
$$

Removing this spatially uniform phase leaves the density and probability current unchanged. Absolute phase colour is a display convention, not an observable. Phase is undefined at wavefunction nodes. The palette is not a calibrated colour bar.

Model time and video time $`t`$ are related by $`\tau=0.28t`$. The 90-second film spans 25.2 model-time units. Camera orbit, lighting, additive line blending, line fading, bloom, and tone mapping are display choices. Pixel brightness is not a quantitative probability-density measurement. The film remains silent unless audio is added separately.

## 7. Validation and limits

`--self-test` checks the analytic overlap norm at four model times and evaluates a position-space density quadrature on a $`96^3`$ grid spanning $`[-7,7]^3`$ at two times. These checks test normalization, not every aspect of the implementation. The completed MP4 is decoded in full to detect encoding errors.

The wavefunction uses double precision in Python and single precision in the GPU shaders. Streamline and surface discretization, geometric seed selection, and display effects affect the visible result. NVIDIA performance depends on the local driver and device configuration. Increasing `--steps` improves curve integration resolution but does not change the analytic quantum state.

## References

- Coherent states: [Georgetown University graduate quantum mechanics notes](https://site.physics.georgetown.edu/~jkf/grad_quant2/html/lecture4.html).
- Probability conservation: [Likharev, Essential Graduate Physics, continuity equation](https://phys.libretexts.org/Bookshelves/Quantum_Mechanics/Essential_Graduate_Physics_-_Quantum_Mechanics_(Likharev)/01%3A_Introduction/1.04%3A_Continuity_Equation).

The equations above are the specification of the implemented model. The numerical parameters come directly from the source code.
