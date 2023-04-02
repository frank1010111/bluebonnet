# Background Equations

Flow into hydraulically fractured wells is primarily one-dimensional. It
responds to pressure gradients in accordance with Darcy's law. The
one-dimensional flow units can be described with a characteristic size.
Sometimes, the easy solution is the right one.

## Single phase flow

Now, if there's only one flowing phase, the equation for mass conservation is

$$
\frac{\partial}{\partial t} \left( \rho \phi S\right) = \frac{\partial}{\partial x}\left( \rho \frac{k}{\mu} \frac{\partial p}{\partial x}\right)
$$

where $\rho$ is density, $S$ is saturation, $x$ is distance from the fracture
face, $t$ is time, $k$ is absolute permeability, and $\mu$ is viscosity. Next,
we simplify things with the pseudopressure transform,

$$
m = \int_{p_{ref}}^p \frac{p}{\mu z}dp
$$

With a few scaling factors, such as $\tilde x = x/d$ to redefine the distance,
$\tilde t= t/\tau$ for time, and $\tilde m = m / m_i$, things simplify.

Then, single phase flow is

$$
\frac{\partial \tilde m}{\partial \tilde t} = \frac{\alpha}{\alpha_i}\frac{\partial^2 \tilde m}{\partial \tilde x^2}
$$

where $d$ is half the distance between adjacent flow units, $\tau$ is the
time-to-boundary dominated flow, and the subscript $i$ indicates the quantity at
the initial reservoir pressure.

The full system of equations is

$$
\begin{cases}
\frac{\partial \tilde m}{\partial \tilde t} &= \frac{\alpha}{\alpha_i}\frac{\partial^2 \tilde m}{\partial \tilde x^2} \\
\tilde m(\tilde x, \tilde t = 0) &= \tilde m_i \\
\tilde m(\tilde x=0, \tilde t) &= \tilde m_f \\
\partial\tilde m/\partial \tilde x |_{\tilde x=1} &= 0
\end{cases}
$$

After solving the system of equations, production can be calculated by applying
Darcy's law at the fracture face.

$$
q = \frac{\mathcal M}{2\tau} \left.\frac{\partial \tilde m}{\partial \tilde t}\right|_{\tilde x=0}
$$

## Simplified two-phase flow

Now, if fluid saturation only varies with pressure, this equation is all that
needs to be solved. The hydraulic diffusivity equation for a multiphase system
is $\alpha=\lambda / c$, where

$$
\begin{align}
% \lambda &= k \left( R_v\frac{k_{rg}}{\mu_g b_g} + \frac{k_{ro}}{\mu_o b_o}\right)\left(\frac{\rho_{o,std}}{\rho_{ref}} + \frac{\rho_{g,std}}{\rho_{ref}} R + \frac{\rho_{w,std}}{\rho_{ref}} W\right)\\
% R &= \frac{\frac{k_{rg}}{\mu_g B_g}+R_s\frac{k_{ro}}{\mu_o B_o}}{R_v\frac{k_{rg}}{\mu_g B_g}+\frac{k_{ro}}{\mu_o B_o}}
% \qquad\qquad\qquad
% W = \frac{k_{rw}/\left(\mu_w b_w\right)}{R_v\frac{k_{rg}}{\mu_g b_g} + \frac{k_ro}{\mu_o b_o}}
% \\
\lambda &= \frac{k}{\rho_{ref}}\left[
    \left(R_v\frac{k_{rg}}{\mu_g b_g} + \frac{k_{ro}}{\mu_o b_o} \right)\rho_{o,std}
    + \left(\frac{k_{rg}}{\mu_g B_g}+R_s\frac{k_{ro}}{\mu_o B_o} \right)\rho_{g,std}
    + \frac{k_{rw}}{\mu_w b_w}\rho_{w,std}\right]
\\
c &= \frac{\partial}{\partial p}\left[\phi\left[
    \frac{\rho_{o,std}}{\rho_{ref}} \left(R_v\frac{S_g}{b_g} + \frac{S_o}{b_o}\right)
    + \frac{\rho_{g,std}}{\rho_{ref}} \left(R_s\frac{S_o}{b_o} + \frac{S_g}{b_o}\right)
    +\frac{\rho_{w,std}}{\rho_{ref}}\frac{S_w}{b_w}\right]
\right]
\end{align}
$$

where $\rho_{j,std}$ is the density of phase $j$ at standard temperature and
pressure, $k_{rj}$ is the relative permeability of phase $j$, $b_j$ is the
formation volume factor of phase $j$, $R_v$ is the volume of oil dissolved into
the gas, and $R_s$ is the volume of gas dissolved in the oil.

A multiphase pseudopressure also looks a little different.

$$
m = \int_{p_ref}^p k\left[
    \frac{\rho_{o,std}}{\rho_{ref}} \left(R_v\frac{k_{rg}}{\mu_g b_g} + \frac{k_{ro}}{\mu_o b_o}\right)
    + \frac{\rho_{g,std}}{\rho_{ref}} \left(R_s \frac{k_{ro}}{\mu_o b_o} + \frac{k_{rg}}{\mu_g b_g}\right)
    + \frac{\rho_{w,std}}{\rho_{ref}} \frac{k_{rw}}{\mu_w b_w}
\right] dp'
$$

## Full three-phase flow (not yet implemented)

Next, full three phases. For water, mass conservation looks like

$$
\begin{equation}
\frac{\partial}{\partial t} \left( \frac{\rho_{w,std}}{\rho_{ref}} \phi \frac{S_w}{b_w}\right) = \frac{\partial}{\partial x}\left( \frac{\rho_{w,std}}{\rho_{ref}} k \frac{k_{rw}}{\mu_w b_w} \frac{\partial p}{\partial x}\right)
\end{equation}
$$

Oil and gas are a bit more complicated, because of miscibility. For natural gas,
the mass conservation equation is

$$
\begin{equation}
\frac{\partial}{\partial t} \left[ \frac{\rho_{g,std}}{\rho_{ref}} \phi \left(R_s\frac{S_o}{b_o} + \frac{S_g}{b_g}\right)\right]
= \frac{\partial}{\partial x}\left[
    \frac{\rho_{g,std}}{\rho_{ref}} k \left(R_s\frac{k_{ro}}{\mu_o b_o} +\frac{k_{rg}}{\mu_g b_g} \frac{\partial p}{\partial x}\right)\right]
\end{equation}
$$

and for oil

$$
\begin{equation}
\frac{\partial}{\partial t} \left[ \frac{\rho_{o,std}}{\rho_{ref}} \phi \left(R_v\frac{S_g}{b_g} + \frac{S_o}{b_o}\right)\right]
= \frac{\partial}{\partial x}\left[
    \frac{\rho_{o,std}}{\rho_{ref}} k \left(R_v\frac{k_{rg}}{\mu_g b_g} +\frac{k_{ro}}{\mu_o b_o} \frac{\partial p}{\partial x}\right)\right]
\end{equation}
$$

<!--\label{oil-sat} -->

The saturation path gets quite complicated.
[Walsh and Lake (2003)](https://www.elsevier.com/books/a-generalized-approach-to-primary-hydrocarbon-recovery-of-petroleum-exploration-and-production/walsh/978-0-444-50683-2)
use an unsteady-state method for calculating the saturation path during primary
recovery. The differential equation is

$$
\begin{align}
\frac{d p}{d S} =
% numerator
\frac{\left(\frac{k{rg}}{\mu_g B_g} +R_s\frac{k{ro}}{\mu_o B_o}\right) \dfrac{\partial \left(R_v\frac{S_g}{B_g} +\frac{S_o}{B_o} \right) }{\partial p}
-\left(R_v\frac{k_{rg}}{\mu_g B_g}+\frac{k_{ro}}{\mu_o B_o}\right)
\dfrac{\partial \left(\frac{S_g}{B_g} +R_s\frac{S_o}{B_o} \right)}{\partial p}}
{\left(R_v\frac{k_{rg}}{\mu_g B_g} +\frac{k{ro}}{\mu_o B_o}\right)
\dfrac{\partial \left(\frac{S_g}{B_g}+R_s\frac{S_o}{B_o} \right)}{\partial S}
-\left(\frac{k{rg}}{\mu_g B_g} +R_s\frac{k{ro}}{\mu_o B_o}\right)
\dfrac{\partial \left(R_v\frac{S_g}{B_g}+\frac{S_o}{B_o} \right) }{\partial S}}
\end{align}
$$

A few other options include an analytical approximation
([Tabatabaie and Pooladi-Darvish, 2017](https://doi.org/10.2118/180932-PA))

$$
\begin{equation}
 So = S_{o,i}+\left(\mu_oB_g\dfrac{\partial R_s}{\partial p}\right){p_{bubble}}\int^{p}{p_{bubble}}\frac{1}{\mu_oB_o}dp'
\end{equation}
$$

and the Constant Composition Expansion Method (CCE)

$$
\begin{equation}
So = 1/\left[{1+\frac{b_g}{b_o}\left(R_{s,i} -R_s\right)}\right]
\end{equation}
$$
