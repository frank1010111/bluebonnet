# Background Equations

Flow into hydraulically fractured wells is primarily one-dimensional. It
responds to pressure gradients in accordance with Darcy's law. The
one-dimensional flow units can be described with a characterestic size.
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

$$\begin{cases}
\frac{\partial \tilde m}{\partial \tilde t} &= \frac{\alpha}{\alpha_i}\frac{\partial^2 \tilde m}{\partial \tilde x^2} \\
\tilde m(\tilde x, \tilde t = 0) &= \tilde m_i \\
\tilde m(\tilde x=0, \tilde t) &= \tilde m_f \\
\partial\tilde m/\partial \tilde x |_{\tilde x=1} &= 0
\end{cases}$$

After solving the system of equations, production can be calculated by applying
Darcy's law at the fracture face.

$$
q = \frac{\mathcal M}{2\tau} \left.\frac{\partial \tilde m}{\partial \tilde t}\right|_{\tilde x=0}
$$

## Simplified two-phase flow

Now, if fluid saturation only varies with pressure, this equation is all that
needs to be solved. The hydraulic diffusivity equation for a multiphase system
is $\alpha=\lambda / c$, where

$$\begin{align}
\lambda &= k \left( R_v\frac{k_{rg}}{\mu_g b_g} + \frac{k_{ro}}{\mu_o b_o}\right)\left(\rho_{o,std} + \rho_{g,std} R + \rho_{w,std} W\right)\\
c &= \frac{\partial}{\partial p}\left\\{\phi\left[
    \rho_{o,std}\left(R_v\frac{S_g}{b_g} + \frac{S_o}{b_o}\right)
    + \rho_{g,std}\left(R_s\frac{S_o}{b_o} + \frac{S_g}{b_o}\right)
    + \rho_{w,std}\frac{S_w}{b_w}\right\\}
\right]
\end{align}$$

where $\rho_{j,std}$ is the density of phase $j$ at standard temperature and
pressure, $k_{rj}$ is the relative permeability of phase $j$, $b_j$ is the
formation volume factor of phase $j$, $R_v$ is the volume of oil dissolved into
the gas, and $R_s$ is the volume of gas dissolved in the oil.

## Full three-phase flow (not yet implemented)

Next, full three phases. For water, mass conservation looks like

$$\begin{equation}
\frac{\partial}{\partial t} \left( \rho_{w,std} \phi \frac{S_w}{b_w}\right) = \frac{\partial}{\partial x}\left( \rho_{w,std} k \frac{k_{rw}}{\mu_w b_w} \frac{\partial p}{\partial x}\right)
\end{equation}$$

Oil and gas are a bit more complicated, because of miscibility. For natural gas,
the mass conservation equation is

$$\begin{equation}
\frac{\partial}{\partial t} \left[ \rho_{g,std} \phi \left(R_s\frac{S_o}{b_o} + \frac{S_g}{b_g}\right)\right]
= \frac{\partial}{\partial x}\left[
    \rho_{g,std} k \left(R_s\frac{k_{ro}}{\mu_o b_o} +\frac{k_{rg}}{\mu_g b_g} \frac{\partial p}{\partial x}\right)\right]
\end{equation}$$

and for oil

$$\begin{equation}
\frac{\partial}{\partial t} \left[ \rho_{o,std} \phi \left(R_v\frac{S_g}{b_g} + \frac{S_o}{b_o}\right)\right]
= \frac{\partial}{\partial x}\left[
    \rho_{o,std} k \left(R_v\frac{k_{rg}}{\mu_g b_g} +\frac{k_{ro}}{\mu_o b_o} \frac{\partial p}{\partial x}\right)\right]
\end{equation}$$

<!--\label{oil-sat} -->

The saturation path gets quite complicated. Let's start defining things.

$$\begin{align}
b &= R_s S_o / b_o + S_g / b_g \\
a &= R_s\frac{k_{ro}}{\mu_o b_o} + \frac{k_{rg}}{\mu_g b_g}
\end{align}$$

Thus, gas saturation follows

$$\begin{equation}
\frac{\partial b}{\partial t} = \frac{k}{\phi} \frac{\partial}{\partial x}\left( a \frac{\partial p}{\partial x}\right)
\end{equation}$$

Oil saturation follows

$$
\begin{align}
\beta &= R_v S_g / b_g + S_o / b_o \\
\alpha &= R_v\frac{k_{rg}}{\mu_g b_g} + \frac{k_{ro}}{\mu_o b_o} \\
\frac{\partial \beta}{\partial t} &= \frac{k}{\phi} \frac{\partial}{\partial x}\left( \alpha \frac{\partial p}{\partial x}\right)
\end{align}
$$

For water,

$$\begin{align}
\xi &= S_w/b_w \\
\gamma &= \frac{k_{rw}}{\mu_w b_w} \\
\frac{\partial \xi}{\partial t} &= \frac{k}{\phi} \frac{\partial}{\partial x}\left( \gamma \frac{\partial p}{\partial x}\right)
\end{align}$$

Next, perform a Boltzman Transform with these convenient partial derivatives

$$\begin{align}
\eta &= x\sqrt{\frac{\phi}{kt}}\\
\frac{\partial\eta}{\partial x} &= \sqrt{\frac{\phi}{kt}}\\
\frac{\partial\eta}{\partial t} &= -\frac{x}{2t}\sqrt{\frac{\phi}{kt}}
\end{align}$$

The saturation equations can be rewritten

$$
\begin{align}
-\frac\eta2 \frac{db}{d\eta} &= \frac{da}{d\eta}\frac{dp}{d\eta} + a \frac{d^2p}{d\eta^2} \\
-\frac\eta2 \frac{d\beta}{d\eta} &= \frac{d\alpha}{d\eta}\frac{dp}{d\eta} + /alpha \frac{d^2p}{d\eta^2} \\
-\frac\eta2 \frac{d\xi}{d\eta} &= \frac{d\gamma}{d\eta}\frac{dp}{d\eta} + \gamma \frac{d^2p}{d\eta^2} \\
\end{align}
$$

where the total derivative for component $Y\in\{a,b,\alpha,\beta,\xi,\gamma\}$
is

$$\begin{equation}
\frac{dY}{d\eta} = \frac{\partial Y}{\partial p}\frac{\partial p}{\partial \eta} + 
\frac{\partial Y}{\partial S_w}\frac{\partial S_w}{\partial \eta} + 
\frac{\partial Y}{\partial S_g}\frac{\partial S_g}{\partial \eta}
\end{equation}$$
