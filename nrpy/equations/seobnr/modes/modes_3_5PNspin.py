"""
Supplementary file to "Spin contributions to the gravitational-waveform modes
for spin-aligned binaries at the 3.5PN order"

Authors: Quentin Henry, Sylvain Marsat, Mohammed Khalil

This file contains the spin-weighted spherical harmonics modes with spin
contributions to 3.5PN order, for circular orbits and aligned spins
(Sec. IV. A in the paper).

Notation:
  nu    = m1*m2 / M        symmetric mass ratio
  delta = (m1 - m2) / M    antisymmetric mass ratio
  M     = m1 + m2          total mass
  x     = (G*M*omega/c**3)**(2/3)  where omega is the orbital frequency
  psi   = phase defined in Eq. (4.1)
  R     = distance from observer to source
  S     = S1 + S2
  Sigma = M/m2*S2 - M/m1*S1
  (kappa1, kappa2, lambda1, lambda2) = spin multipole constants (= 1 for BHs)
  kappap = kappa1 + kappa2,  kappam = kappa1 - kappa2
  lambdap = lambda1 + lambda2, lambdam = lambda1 - lambda2
"""

from sympy import (
    symbols, sqrt, pi, exp, log, I, Rational, acoth,
)

# ── Symbols ──────────────────────────────────────────────────────────────────
G, M, c, R = symbols('G M c R', positive=True)
x, nu, delta, psi = symbols('x nu delta psi')
S, Sigma = symbols('S Sigma')
kappap, kappam = symbols('kappa_p kappa_m')
lambdap, lambdam = symbols('lambda_p lambda_m')

# ── Helper ────────────────────────────────────────────────────────────────────
# ArcCoth[5] in Mathematica = acoth(5)
ArcCoth5 = acoth(5)

# Common prefactor  8*G*M*sqrt(pi/5)*x*nu / (c^2 * R)
def _pre(m_mode):
    """Return the common prefactor with the exponential e^{-i*m*psi}."""
    return 8*G*M*sqrt(pi/5)*x*nu / (c**2 * exp(I*m_mode*psi) * R)

# ── Mode h[2,1] ───────────────────────────────────────────────────────────────
h21 = _pre(1) * (
    (I/3)*sqrt(x)*delta
    + x**Rational(3,2)*delta*(Rational(-17,84)*I + Rational(5,21)*I*nu)
    + (I/2)*x*Sigma/(G*M**2)
    + x**3*(
        (S*delta*(Rational(-331,756)*I + Rational(386,189)*I*nu)
         + (Rational(293,756)*I - Rational(2615,1512)*I*nu - Rational(1723,378)*I*nu**2)*Sigma
        )/(G*M**2)
        + (S**2*(I/2 + I/4*kappap)*Sigma
           + S*((I/2)*delta - I/4*kappam + I/4*delta*kappap)*Sigma**2
           + ((-I/8)*delta*kappam + kappap*(I/8 - I/4*nu) - I/2*nu)*Sigma**3
          )/(G**3*M**6)
    )
    + x**Rational(5,2)*(
        (S**2*(I*delta - I/3*kappam + I/2*delta*kappap)
         + S*(-I/3 - Rational(5,6)*I*delta*kappam
               + kappap*(Rational(5,6)*I - 2*I*nu) - 4*I*nu)*Sigma
         + (delta*kappap*(Rational(5,12)*I - I/2*nu)
            + delta*(-I/2 - I*nu)
            + kappam*(Rational(-5,12)*I + Rational(4,3)*I*nu)
           )*Sigma**2
        )/(G**2*M**4)
        + Sigma*(Rational(1,4) + I/2*pi + log(2))/(G*M**2)
    )
    + x**Rational(7,2)*(
        (S**2*(delta*kappap*(Rational(47,336)*I - I/14*nu)
               + delta*(Rational(41,42)*I - I/7*nu)
               + kappam*(Rational(23,48)*I - Rational(191,72)*I*nu))
         + S*(Rational(-29,21)*I
              + delta*kappam*(Rational(19,56)*I - Rational(1301,504)*I*nu)
              + Rational(100,21)*I*nu + Rational(4,7)*I*nu**2
              + kappap*(Rational(-19,56)*I + Rational(1019,504)*I*nu + Rational(2,7)*I*nu**2)
             )*Sigma
         + (delta*kappap*(Rational(-19,112)*I + Rational(145,126)*I*nu + I/14*nu**2)
            + delta*(Rational(-6,7)*I + Rational(59,21)*I*nu + I/7*nu**2)
            + kappam*(Rational(19,112)*I - Rational(751,504)*I*nu + Rational(1265,504)*I*nu**2)
           )*Sigma**2
        )/(G**2*M**4)
        + (S*delta*(-Rational(181,210) - Rational(43,21)*I*pi - Rational(86,21)*log(2))
           + Sigma*(-Rational(79,84) - Rational(79,42)*I*pi - Rational(79,21)*log(2)
                    + nu*(Rational(1951,280) + Rational(257,84)*I*pi + Rational(257,42)*log(2)))
          )/(G*M**2)
    )
    + x**2*(
        (Rational(-43,21)*I*S*delta
         + (Rational(-79,42)*I + Rational(139,42)*I*nu)*Sigma
        )/(G*M**2)
        + delta*(Rational(1,6) + I/3*pi + log(16)/6)
    )
)

# ── Mode h[2,2] ───────────────────────────────────────────────────────────────
h22 = _pre(2) * (
    1
    + x*(Rational(-107,42) + Rational(55,42)*nu)
    + x**Rational(5,2)*(
        S*(Rational(-163,63) - Rational(92,63)*nu)
        + delta*(Rational(-1,21) + Rational(20,63)*nu)*Sigma
    )/(G*M**2)
    + x**Rational(3,2)*(
        2*pi
        + (-2*S - Rational(2,3)*delta*Sigma)/(G*M**2)
    )
    + x**2*(
        Rational(-2173,1512) - Rational(1069,216)*nu + Rational(2047,1512)*nu**2
        + (S**2*(2 + kappap)
           + S*(2*delta - kappam + delta*kappap)*Sigma
           + (-Rational(1,2)*delta*kappam + kappap*(Rational(1,2) - nu) - 2*nu)*Sigma**2
          )/(G**2*M**4)
    )
    + x**3*(
        ((Rational(-4,3)*I - 4*pi)*S - 4*pi*delta*Sigma/3)/(G*M**2)
        + (S**2*(Rational(-404,63) + Rational(55,42)*delta*kappam
                 + Rational(68,21)*nu + kappap*(Rational(-31,42) + Rational(34,21)*nu))
           + S*(kappam*(Rational(43,21) - Rational(48,7)*nu)
                + delta*kappap*(Rational(-43,21) + Rational(34,21)*nu)
                + delta*(Rational(-481,63) + Rational(68,21)*nu)
               )*Sigma
           + (Rational(-5,3)
              + delta*kappam*(Rational(43,42) - Rational(89,42)*nu)
              + Rational(172,21)*nu - Rational(68,21)*nu**2
              + kappap*(Rational(-43,42) + Rational(25,6)*nu - Rational(34,21)*nu**2)
             )*Sigma**2
          )/(G**2*M**4)
    )
    + x**Rational(7,2)*(
        (S*(Rational(1061,84) + Rational(4043,84)*nu + Rational(499,84)*nu**2)
         + delta*(Rational(3931,756) + Rational(7813,378)*nu + Rational(1025,252)*nu**2)*Sigma
        )/(G*M**2)
        + (S**2*(4*pi + 2*pi*kappap)
           + S*(4*pi*delta - 2*pi*kappam + 2*pi*delta*kappap)*Sigma
           + (-pi*delta*kappam - 4*pi*nu + kappap*(pi - 2*pi*nu))*Sigma**2
          )/(G**2*M**4)
        + (S**3*(Rational(32,3) - Rational(2,3)*kappap - 2*lambdap)
           + S**2*(Rational(-7,3)*kappam - Rational(1,3)*delta*kappap
                   + 3*lambdam + delta*(Rational(52,3) - 3*lambdap))*Sigma
           + S*(Rational(20,3) - 3*delta*kappam + 3*delta*lambdam - 3*lambdap
                + kappap*(3 - Rational(2,3)*nu)
                + (-Rational(112,3) + 6*lambdap)*nu
               )*Sigma**2
           + (lambdam + delta*kappap*(Rational(5,3) - nu/3)
              - 3*lambdam*nu + kappam*(Rational(-5,3) + Rational(11,3)*nu)
              + delta*(-lambdap + (-Rational(20,3) + lambdap)*nu)
             )*Sigma**3
          )/(G**3*M**6)
    )
)

# ── Mode h[3,1] ───────────────────────────────────────────────────────────────
h31 = _pre(1) * (
    I/12*sqrt(x)*delta/sqrt(14)
    + x**Rational(3,2)*delta*(-I/9*sqrt(Rational(2,7)) - I/18*nu/sqrt(14))
    + x**3*(
        S*delta*(Rational(-79,216)*I/sqrt(14) + Rational(443,216)*I*nu/sqrt(14))
        + (Rational(-149,216)*I/sqrt(14) + Rational(25,54)*I*sqrt(Rational(7,2))*nu
           - Rational(841,216)*I*nu**2/sqrt(14))*Sigma
    )/(G*M**2)
    + x**Rational(5,2)*(
        S**2*(I/4*delta/sqrt(14) - I/3*kappam/sqrt(14) + I/8*delta*kappap/sqrt(14))
        + S*(I/4/sqrt(14) - Rational(11,24)*I*delta*kappam/sqrt(14) - I*nu/sqrt(14)
             + kappap*(Rational(11,24)*I/sqrt(14) - I/2*nu/sqrt(14)))*Sigma
        + ((-I/4)*delta*nu/sqrt(14)
           + kappam*(Rational(-11,48)*I/sqrt(14) + I/12*sqrt(Rational(7,2))*nu)
           + delta*kappap*(Rational(11,48)*I/sqrt(14) - I/8*nu/sqrt(14))
          )*Sigma**2
    )/(G**2*M**4)
    + x**2*(
        (I/24*S*delta/sqrt(14) + (Rational(5,24)*I/sqrt(14) - Rational(5,8)*I*nu/sqrt(14))*Sigma)/(G*M**2)
        + delta*(sqrt(Rational(7,2))/60 + I/12*pi/sqrt(14) + log(1024)/(60*sqrt(14)))
    )
    + x**Rational(7,2)*(
        (S**2*(kappam*(Rational(13,16)*I/sqrt(14) - Rational(2,9)*I*sqrt(Rational(2,7))*nu)
               + delta*kappap*(Rational(-53,144)*I/sqrt(14) - Rational(11,36)*I*nu/sqrt(14))
               + delta*(Rational(-149,72)*I/sqrt(14) - Rational(11,18)*I*nu/sqrt(14)))
         + S*(Rational(-115,36)*I/sqrt(14) + 5*I*sqrt(Rational(2,7))*nu
              + Rational(11,9)*I*sqrt(Rational(2,7))*nu**2
              + delta*kappam*(Rational(85,72)*I/sqrt(14) - Rational(5,36)*I*nu/sqrt(14))
              + kappap*(Rational(-85,72)*I/sqrt(14) + Rational(29,18)*I*nu/sqrt(14)
                        + Rational(11,9)*I*nu**2/sqrt(14))
             )*Sigma
         + (kappam*(Rational(85,144)*I/sqrt(14) - Rational(233,144)*I*nu/sqrt(14)
                    - I/6*nu**2/sqrt(14))
            + delta*kappap*(Rational(-85,144)*I/sqrt(14) + I/16*sqrt(Rational(7,2))*nu
                            + Rational(11,36)*I*nu**2/sqrt(14))
            + delta*(Rational(-9,8)*I/sqrt(14) + Rational(25,9)*I*nu/sqrt(14)
                     + Rational(11,18)*I*nu**2/sqrt(14))
           )*Sigma**2
        )/(G**2*M**4)
        + (S*delta*(-Rational(47,240)/sqrt(14) + I/24*pi/sqrt(14) + log(2)/(12*sqrt(14)))
           + Sigma*(sqrt(Rational(7,2))/24
                    + Rational(5,24)*I*pi/sqrt(14)
                    + nu*(-Rational(11,240)/sqrt(14) - Rational(5,8)*I*pi/sqrt(14)
                          - Rational(5,4)*log(2)/sqrt(14))
                    + log(1024)/(24*sqrt(14)))
          )/(G*M**2)
    )
)

# ── Mode h[3,2] ───────────────────────────────────────────────────────────────
h32 = _pre(2) * (
    x*(sqrt(Rational(5,7))/3 - sqrt(Rational(5,7))*nu)
    + x**2*(-Rational(193,54)/sqrt(35) + Rational(145,54)*sqrt(Rational(5,7))*nu
            - Rational(73,54)*sqrt(Rational(5,7))*nu**2)
    + x**Rational(3,2)*(2*sqrt(Rational(5,7))*S + 2*sqrt(Rational(5,7))*delta*Sigma/3)/(G*M**2)
    + x**Rational(5,2)*(
        S*(Rational(-13,3)*sqrt(Rational(5,7)) + Rational(73,9)*sqrt(Rational(5,7))*nu)
        + delta*(Rational(-31,9)*sqrt(Rational(5,7)) + Rational(10,3)*sqrt(Rational(5,7))*nu)*Sigma
    )/(G*M**2)
    + x**3*(
        ((Rational(-2,3)*I + Rational(4,3)*pi)*sqrt(Rational(5,7))*S
         + (-2*I + Rational(4,3)*pi)*sqrt(Rational(5,7))*delta*Sigma
        )/(G*M**2)
        + (S**2*(Rational(-8,9)*sqrt(Rational(5,7)) - Rational(1,3)*sqrt(Rational(5,7))*delta*kappam
                 - 4*sqrt(Rational(5,7))*nu + kappap*(sqrt(Rational(5,7)) - 2*sqrt(Rational(5,7))*nu))
           + S*(delta*(Rational(-20,9)*sqrt(Rational(5,7)) - 4*sqrt(Rational(5,7))*nu)
                + delta*kappap*(Rational(4,3)*sqrt(Rational(5,7)) - 2*sqrt(Rational(5,7))*nu)
                + kappam*(Rational(-4,3)*sqrt(Rational(5,7)) + Rational(10,3)*sqrt(Rational(5,7))*nu)
               )*Sigma
           + (Rational(-4,3)*sqrt(Rational(5,7)) + Rational(8,3)*sqrt(Rational(5,7))*nu
              + 4*sqrt(Rational(5,7))*nu**2
              + delta*kappam*(Rational(-2,3)*sqrt(Rational(5,7)) + Rational(4,3)*sqrt(Rational(5,7))*nu)
              + kappap*(Rational(2,3)*sqrt(Rational(5,7)) - Rational(8,3)*sqrt(Rational(5,7))*nu
                        + 2*sqrt(Rational(5,7))*nu**2)
             )*Sigma**2
          )/(G**2*M**4)
    )
    + x**Rational(7,2)*(
        (S*(Rational(4859,396)/sqrt(35) - Rational(15413,1188)*sqrt(Rational(5,7))*nu
            - Rational(419,132)*sqrt(Rational(5,7))*nu**2)
         + delta*(Rational(19241,1188)/sqrt(35) - Rational(1616,33)/sqrt(35)*nu
                  - Rational(16153,1188)/sqrt(35)*nu**2)*Sigma
        )/(G*M**2)
        + (S**3*(Rational(4,3)*sqrt(Rational(5,7)) + Rational(2,3)*sqrt(Rational(5,7))*kappap)
           + S**2*(Rational(8,3)*sqrt(Rational(5,7))*delta - Rational(2,3)*sqrt(Rational(5,7))*kappam
                   + Rational(4,3)*sqrt(Rational(5,7))*delta*kappap)*Sigma
           + S*(Rational(4,3)*sqrt(Rational(5,7)) - sqrt(Rational(5,7))*delta*kappam
                - Rational(20,3)*sqrt(Rational(5,7))*nu
                + kappap*(sqrt(Rational(5,7)) - Rational(10,3)*sqrt(Rational(5,7))*nu)
               )*Sigma**2
           + (Rational(-4,3)*sqrt(Rational(5,7))*delta*nu
              + delta*kappap*(Rational(1,3)*sqrt(Rational(5,7)) - Rational(2,3)*sqrt(Rational(5,7))*nu)
              + kappam*(-Rational(1,3)*sqrt(Rational(5,7)) + Rational(4,3)*sqrt(Rational(5,7))*nu)
             )*Sigma**3
          )/(G**3*M**6)
    )
)

# ── Mode h[3,3] ───────────────────────────────────────────────────────────────
h33 = _pre(3) * (
    Rational(-3,4)*I*sqrt(Rational(15,14))*sqrt(x)*delta
    + x**Rational(3,2)*delta*(3*I*sqrt(Rational(15,14)) - Rational(3,2)*I*sqrt(Rational(15,14))*nu)
    + x**3*(
        S*delta*(Rational(-139,8)*I*sqrt(Rational(3,70)) + Rational(83,8)*I*sqrt(Rational(3,70))*nu)
        + (Rational(-129,8)*I*sqrt(Rational(3,70)) + 9*I*sqrt(Rational(15,14))*nu
           + Rational(15,8)*I*sqrt(Rational(15,14))*nu**2)*Sigma
    )/(G*M**2)
    + x**Rational(5,2)*(
        S**2*(Rational(-9,4)*I*sqrt(Rational(15,14))*delta - Rational(9,8)*I*sqrt(Rational(15,14))*delta*kappap)
        + S*(Rational(-9,4)*I*sqrt(Rational(15,14)) + Rational(9,8)*I*sqrt(Rational(15,14))*delta*kappam
             + 9*I*sqrt(Rational(15,14))*nu
             + kappap*(Rational(-9,8)*I*sqrt(Rational(15,14)) + Rational(9,2)*I*sqrt(Rational(15,14))*nu)
            )*Sigma
        + (Rational(9,4)*I*sqrt(Rational(15,14))*delta*nu
           + delta*kappap*(Rational(-9,16)*I*sqrt(Rational(15,14)) + Rational(9,8)*I*sqrt(Rational(15,14))*nu)
           + kappam*(Rational(9,16)*I*sqrt(Rational(15,14)) - Rational(9,4)*I*sqrt(Rational(15,14))*nu)
          )*Sigma**2
    )/(G**2*M**4)
    + x**2*(
        (Rational(3,8)*I*sqrt(Rational(105,2))*S*delta
         + (Rational(9,8)*I*sqrt(Rational(15,14)) - Rational(27,8)*I*sqrt(Rational(15,14))*nu)*Sigma
        )/(G*M**2)
        + delta*(-Rational(9,4)*sqrt(Rational(21,10)) - Rational(9,4)*I*sqrt(Rational(15,14))*pi
                 + Rational(9,4)*sqrt(Rational(3,70))*log(Rational(59049,1024)))
    )
    + x**Rational(7,2)*(
        (S**2*(delta*kappap*(Rational(45,16)*I*sqrt(Rational(15,14)) - Rational(9,4)*I*sqrt(Rational(15,14))*nu)
               + delta*(Rational(69,8)*I*sqrt(Rational(15,14)) - Rational(9,2)*I*sqrt(Rational(15,14))*nu)
               + kappam*(Rational(-3,16)*I*sqrt(Rational(105,2)) + 3*I*sqrt(Rational(30,7))*nu))
         + S*(Rational(39,4)*I*sqrt(Rational(15,14)) - 24*I*sqrt(Rational(30,7))*nu
              + 9*I*sqrt(Rational(30,7))*nu**2
              + delta*kappam*(Rational(-33,8)*I*sqrt(Rational(15,14)) + Rational(33,4)*I*sqrt(Rational(15,14))*nu)
              + kappap*(Rational(33,8)*I*sqrt(Rational(15,14)) - Rational(39,2)*I*sqrt(Rational(15,14))*nu
                        + 9*I*sqrt(Rational(15,14))*nu**2)
             )*Sigma
         + (delta*kappap*(Rational(33,16)*I*sqrt(Rational(15,14)) - Rational(111,16)*I*sqrt(Rational(15,14))*nu
                          + Rational(9,4)*I*sqrt(Rational(15,14))*nu**2)
            + delta*(Rational(9,8)*I*sqrt(Rational(15,14)) - 6*I*sqrt(Rational(30,7))*nu
                     + Rational(9,2)*I*sqrt(Rational(15,14))*nu**2)
            + kappam*(Rational(-33,16)*I*sqrt(Rational(15,14)) + Rational(177,16)*I*sqrt(Rational(15,14))*nu
                      - Rational(3,2)*I*sqrt(Rational(105,2))*nu**2)
           )*Sigma**2
        )/(G**2*M**4)
        + (S*delta*(Rational(639,16)*sqrt(Rational(3,70)) + Rational(9,8)*I*sqrt(Rational(105,2))*pi
                    - Rational(9,2)*sqrt(Rational(105,2))*ArcCoth5)
           + Sigma*(Rational(27,8)*sqrt(Rational(21,10)) + Rational(27,8)*I*sqrt(Rational(15,14))*pi
                    + nu*(-Rational(8797,48)/sqrt(210) - Rational(81,8)*I*sqrt(Rational(15,14))*pi
                          + Rational(81,2)*sqrt(Rational(15,14))*ArcCoth5)
                    - Rational(27,4)*sqrt(Rational(15,14))*log(3)
                    + Rational(27,8)*sqrt(Rational(3,70))*log(1024))
          )/(G*M**2)
    )
)

# ── Mode h[4,1] ───────────────────────────────────────────────────────────────
h41 = _pre(1) * (
    x**Rational(3,2)*delta*(I/84/sqrt(10) - I/42*nu/sqrt(10))
    + x**2*(I/168*sqrt(Rational(5,2))*S*delta
            + (I/168*sqrt(Rational(5,2)) - I/56*sqrt(Rational(5,2))*nu)*Sigma
           )/(G*M**2)
    + x**3*(
        S*delta*(Rational(-1147,5544)*I/sqrt(10) + Rational(1139,5544)*I*nu/sqrt(10))
        + (Rational(-103,616)*I/sqrt(10) + Rational(29,231)*I*sqrt(Rational(5,2))*nu
           - Rational(37,616)*I*sqrt(Rational(5,2))*nu**2)*Sigma
    )/(G*M**2)
    + x**Rational(7,2)*(
        (S**2*(delta*kappap*(I/14/sqrt(10) - I/84*sqrt(Rational(5,2))*nu)
               + delta*(-I/56*sqrt(Rational(5,2)) - I/42*sqrt(Rational(5,2))*nu)
               + kappam*(-I/24/sqrt(10) + I/12*nu/sqrt(10)))
         + S*((-I/28)*sqrt(Rational(5,2)) + I/14*sqrt(Rational(5,2))*nu
              + I/21*sqrt(10)*nu**2
              + delta*kappam*(Rational(-19,168)*I/sqrt(10) + I/7*nu/sqrt(10))
              + kappap*(Rational(19,168)*I/sqrt(10) - Rational(3,7)*I*nu/sqrt(10)
                        + I/21*sqrt(Rational(5,2))*nu**2)
             )*Sigma
         + (delta*kappap*(Rational(19,336)*I/sqrt(10) - I/7*nu/sqrt(10) + I/84*sqrt(Rational(5,2))*nu**2)
            + delta*(-I/56*sqrt(Rational(5,2)) + I/42*sqrt(Rational(5,2))*nu + I/42*sqrt(Rational(5,2))*nu**2)
            + kappam*(Rational(-19,336)*I/sqrt(10) + Rational(43,168)*I*nu/sqrt(10)
                      - Rational(17,84)*I*nu**2/sqrt(10))
           )*Sigma**2
        )/(G**2*M**4)
        + (S*delta*(Rational(53,1008)/sqrt(10) + I/168*sqrt(Rational(5,2))*pi
                    + sqrt(Rational(5,2))*log(2)/84)
           + Sigma*(2*sqrt(Rational(2,5))/63 + I/168*sqrt(Rational(5,2))*pi
                    + sqrt(Rational(5,2))*log(2)/84
                    + nu*(-Rational(181,1008)/sqrt(10) - I/56*sqrt(Rational(5,2))*pi
                          - sqrt(Rational(5,2))*log(2)/28))
          )/(G*M**2)
    )
)

# ── Mode h[4,2] ───────────────────────────────────────────────────────────────
h42 = _pre(2) * (
    x*(sqrt(5)/63 - sqrt(5)*nu/21)
    + x**2*(-Rational(437,1386)/sqrt(5) + Rational(115,594)*sqrt(5)*nu - Rational(19,1386)*sqrt(5)*nu**2)
    + x**Rational(5,2)*(
        S*(Rational(-4,189)/sqrt(5) + Rational(4,63)*nu/sqrt(5))
        + delta*(Rational(4,21)/sqrt(5) - Rational(8,21)*nu/sqrt(5))*Sigma
    )/(G*M**2)
    + x**Rational(7,2)*(
        S*(Rational(-86,231)/sqrt(5) + Rational(6653,2079)*nu/sqrt(5) - Rational(1387,231)*nu**2/sqrt(5))
        + delta*(Rational(-626,693)/sqrt(5) + Rational(6698,2079)*nu/sqrt(5)
                 - Rational(145,231)*sqrt(5)*nu**2)*Sigma
    )/(G*M**2)
    + x**3*(
        S**2*(Rational(4,63)*sqrt(5) - Rational(1,21)*sqrt(5)*delta*kappam
              - Rational(4,21)*sqrt(5)*nu + kappap*(Rational(5,63)*sqrt(5) - Rational(2,21)*sqrt(5)*nu))
        + S*(delta*(Rational(4,63)*sqrt(5) - Rational(4,21)*sqrt(5)*nu)
             + delta*kappap*(Rational(8,63)*sqrt(5) - Rational(2,21)*sqrt(5)*nu)
             + kappam*(Rational(-8,63)*sqrt(5) + Rational(2,7)*sqrt(5)*nu)
            )*Sigma
        + (Rational(-4,63)*sqrt(5)*nu + Rational(4,21)*sqrt(5)*nu**2
           + delta*kappam*(Rational(-4,63)*sqrt(5) + Rational(2,21)*sqrt(5)*nu)
           + kappap*(Rational(4,63)*sqrt(5) - Rational(2,9)*sqrt(5)*nu + Rational(2,21)*sqrt(5)*nu**2)
          )*Sigma**2
    )/(G**2*M**4)
)

# ── Mode h[4,3] ───────────────────────────────────────────────────────────────
h43 = _pre(3) * (
    x**Rational(3,2)*delta*(Rational(-9,4)*I/sqrt(70) + Rational(9,2)*I*nu/sqrt(70))
    + x**2*(Rational(-9,8)*I*sqrt(Rational(5,14))*S*delta
            + (Rational(-9,8)*I*sqrt(Rational(5,14)) + Rational(27,8)*I*sqrt(Rational(5,14))*nu)*Sigma
           )/(G*M**2)
    + x**3*(
        S*delta*(Rational(3909,88)*I/sqrt(70) - Rational(4353,88)*I*nu/sqrt(70))
        + (Rational(3249,88)*I/sqrt(70) - Rational(639,22)*I*sqrt(Rational(5,14))*nu
           + Rational(1467,88)*I*sqrt(Rational(5,14))*nu**2)*Sigma
    )/(G*M**2)
    + x**Rational(7,2)*(
        (S**2*(delta*kappap*(-9*I/sqrt(70) + Rational(9,4)*I*sqrt(Rational(5,14))*nu)
               + delta*(Rational(27,8)*I*sqrt(Rational(5,14)) + Rational(9,2)*I*sqrt(Rational(5,14))*nu)
               + kappam*(Rational(27,8)*I/sqrt(70) - Rational(27,4)*I*nu/sqrt(70)))
         + S*(Rational(27,4)*I*sqrt(Rational(5,14)) - Rational(27,2)*I*sqrt(Rational(5,14))*nu
              - 9*I*sqrt(Rational(10,7))*nu**2
              + delta*kappam*(Rational(99,8)*I/sqrt(70) - 9*I*sqrt(Rational(2,35))*nu)
              + kappap*(Rational(-99,8)*I/sqrt(70) + 27*I*sqrt(Rational(2,35))*nu
                        - 9*I*sqrt(Rational(5,14))*nu**2)
             )*Sigma
         + (delta*kappap*(Rational(-99,16)*I/sqrt(70) + 9*I*sqrt(Rational(2,35))*nu
                          - Rational(9,4)*I*sqrt(Rational(5,14))*nu**2)
            + delta*(Rational(27,8)*I*sqrt(Rational(5,14)) - Rational(9,2)*I*sqrt(Rational(5,14))*nu
                     - Rational(9,2)*I*sqrt(Rational(5,14))*nu**2)
            + kappam*(Rational(99,16)*I/sqrt(70) - Rational(243,8)*I*nu/sqrt(70)
                      + Rational(117,4)*I*nu**2/sqrt(70))
           )*Sigma**2
        )/(G**2*M**4)
        + (S*delta*(Rational(-477,16)/sqrt(70) - Rational(27,8)*I*sqrt(Rational(5,14))*pi
                    + Rational(27,2)*sqrt(Rational(5,14))*ArcCoth5)
           + Sigma*(-18*sqrt(Rational(2,35)) - Rational(27,8)*I*sqrt(Rational(5,14))*pi
                    + Rational(27,2)*sqrt(Rational(5,14))*ArcCoth5
                    + nu*(Rational(6007,48)/sqrt(70) + Rational(81,8)*I*sqrt(Rational(5,14))*pi
                          - Rational(81,2)*sqrt(Rational(5,14))*ArcCoth5))
          )/(G*M**2)
    )
)

# ── Mode h[4,4] ───────────────────────────────────────────────────────────────
h44 = _pre(4) * (
    x*(Rational(-8,9)*sqrt(Rational(5,7)) + Rational(8,3)*sqrt(Rational(5,7))*nu)
    + x**2*(Rational(2372,99)/sqrt(35) - Rational(5092,297)*sqrt(Rational(5,7))*nu
            + Rational(100,99)*sqrt(35)*nu**2)
    + x**Rational(5,2)*(
        S*(Rational(608,27)/sqrt(35) - Rational(608,9)*nu/sqrt(35))
        + delta*(Rational(32,3)/sqrt(35) - Rational(64,3)*nu/sqrt(35))*Sigma
    )/(G*M**2)
    + x**Rational(7,2)*(
        S*(Rational(-6992,99)/sqrt(35) + Rational(80504,297)*nu/sqrt(35)
           - Rational(7768,99)*nu**2/sqrt(35))
        + delta*(Rational(-544,11)/sqrt(35) + Rational(6928,297)*sqrt(Rational(5,7))*nu
                 + Rational(536,99)*nu**2/sqrt(35))*Sigma
    )/(G*M**2)
    + x**3*(
        S**2*(Rational(-32,9)*sqrt(Rational(5,7)) + Rational(32,3)*sqrt(Rational(5,7))*nu
              + kappap*(Rational(-16,9)*sqrt(Rational(5,7)) + Rational(16,3)*sqrt(Rational(5,7))*nu))
        + S*(kappam*(Rational(16,9)*sqrt(Rational(5,7)) - Rational(16,3)*sqrt(Rational(5,7))*nu)
             + delta*kappap*(Rational(-16,9)*sqrt(Rational(5,7)) + Rational(16,3)*sqrt(Rational(5,7))*nu)
             + delta*(Rational(-32,9)*sqrt(Rational(5,7)) + Rational(32,3)*sqrt(Rational(5,7))*nu)
            )*Sigma
        + (Rational(32,9)*sqrt(Rational(5,7))*nu - Rational(32,3)*sqrt(Rational(5,7))*nu**2
           + delta*kappam*(Rational(8,9)*sqrt(Rational(5,7)) - Rational(8,3)*sqrt(Rational(5,7))*nu)
           + kappap*(Rational(-8,9)*sqrt(Rational(5,7)) + Rational(40,9)*sqrt(Rational(5,7))*nu
                     - Rational(16,3)*sqrt(Rational(5,7))*nu**2)
          )*Sigma**2
    )/(G**2*M**4)
)

# ── Mode h[5,1] ───────────────────────────────────────────────────────────────
h51 = _pre(1) * (
    x**Rational(3,2)*delta*(I/288/sqrt(385) - I/144*nu/sqrt(385))
    + x**3*(
        S*delta*(I/216/sqrt(385) - I/108*nu/sqrt(385))
        + (I/432*sqrt(Rational(7,55)) - I/432*sqrt(Rational(35,11))*nu
           + I/432*sqrt(Rational(35,11))*nu**2)*Sigma
    )/(G*M**2)
    + x**Rational(7,2)*(
        (S**2*(delta*kappap*(Rational(17,576)*I/sqrt(385) - I/288*sqrt(Rational(5,77))*nu)
               + delta*(I/288*sqrt(Rational(5,77)) - I/144*sqrt(Rational(5,77))*nu)
               + kappam*(-I/48/sqrt(385) + I/24*nu/sqrt(385)))
         + S*(I/288*sqrt(Rational(5,77)) - I/48*sqrt(Rational(5,77))*nu
              + I/36*sqrt(Rational(5,77))*nu**2
              + delta*kappam*(Rational(-29,576)*I/sqrt(385) + Rational(17,288)*I*nu/sqrt(385))
              + kappap*(Rational(29,576)*I/sqrt(385) - Rational(17,96)*I*nu/sqrt(385)
                        + I/72*sqrt(Rational(5,77))*nu**2)
             )*Sigma
         + (delta*kappap*(Rational(29,1152)*I/sqrt(385) - Rational(17,288)*I*nu/sqrt(385)
                          + I/288*sqrt(Rational(5,77))*nu**2)
            + delta*(-I/288*sqrt(Rational(5,77))*nu + I/144*sqrt(Rational(5,77))*nu**2)
            + kappam*(Rational(-29,1152)*I/sqrt(385) + I/64*sqrt(Rational(7,55))*nu
                      - I/144*sqrt(Rational(11,35))*nu**2)
           )*Sigma**2
        )/(G**2*M**4)
    )
)

# ── Mode h[5,2] ───────────────────────────────────────────────────────────────
h52 = _pre(2) * (
    x**2*(Rational(2,27)/sqrt(55) - Rational(2,27)*sqrt(Rational(5,11))*nu
          + Rational(2,27)*sqrt(Rational(5,11))*nu**2)
    + x**Rational(5,2)*(
        S*(Rational(2,9)/sqrt(55) - Rational(2,3)*nu/sqrt(55))
        + delta*(Rational(2,9)/sqrt(55) - Rational(4,9)*nu/sqrt(55))*Sigma
    )/(G*M**2)
    + x**Rational(7,2)*(
        S*(Rational(-71,39)/sqrt(55) + Rational(229,351)*sqrt(Rational(11,5))*nu
           - Rational(493,117)*nu**2/sqrt(55))
        + delta*(Rational(-107,351)*sqrt(Rational(5,11)) + Rational(488,117)*nu/sqrt(55)
                 - Rational(113,351)*sqrt(Rational(5,11))*nu**2)*Sigma
    )/(G*M**2)
)

# ── Mode h[5,3] ───────────────────────────────────────────────────────────────
h53 = _pre(3) * (
    x**Rational(3,2)*delta*(Rational(-9,32)*I*sqrt(Rational(3,110))
                            + Rational(9,16)*I*sqrt(Rational(3,110))*nu)
    + x**3*(
        S*delta*(Rational(3,8)*I*sqrt(Rational(3,110)) - Rational(3,4)*I*sqrt(Rational(3,110))*nu)
        + (Rational(-9,16)*I*sqrt(Rational(3,110)) + Rational(9,16)*I*sqrt(Rational(15,22))*nu
           - Rational(9,16)*I*sqrt(Rational(15,22))*nu**2)*Sigma
    )/(G*M**2)
    + x**Rational(7,2)*(
        (S**2*(kappam*(Rational(9,8)*I*sqrt(Rational(3,110)) - Rational(9,4)*I*sqrt(Rational(3,110))*nu)
               + delta*kappap*(Rational(-117,64)*I*sqrt(Rational(3,110))
                               + Rational(9,32)*I*sqrt(Rational(15,22))*nu)
               + delta*(Rational(-9,32)*I*sqrt(Rational(15,22)) + Rational(9,16)*I*sqrt(Rational(15,22))*nu))
         + S*(Rational(-9,32)*I*sqrt(Rational(15,22)) + Rational(27,16)*I*sqrt(Rational(15,22))*nu
              - Rational(9,4)*I*sqrt(Rational(15,22))*nu**2
              + delta*kappam*(Rational(189,64)*I*sqrt(Rational(3,110))
                              - Rational(117,32)*I*sqrt(Rational(3,110))*nu)
              + kappap*(Rational(-189,64)*I*sqrt(Rational(3,110))
                        + Rational(351,32)*I*sqrt(Rational(3,110))*nu
                        - Rational(9,8)*I*sqrt(Rational(15,22))*nu**2)
             )*Sigma
         + (kappam*(Rational(189,128)*I*sqrt(Rational(3,110)) - Rational(423,64)*I*sqrt(Rational(3,110))*nu
                    + Rational(81,16)*I*sqrt(Rational(3,110))*nu**2)
            + delta*kappap*(Rational(-189,128)*I*sqrt(Rational(3,110))
                            + Rational(117,32)*I*sqrt(Rational(3,110))*nu
                            - Rational(9,32)*I*sqrt(Rational(15,22))*nu**2)
            + delta*(Rational(9,32)*I*sqrt(Rational(15,22))*nu
                     - Rational(9,16)*I*sqrt(Rational(15,22))*nu**2)
           )*Sigma**2
        )/(G**2*M**4)
    )
)

# ── Mode h[5,4] ───────────────────────────────────────────────────────────────
h54 = _pre(4) * (
    x**2*(Rational(-32,9)/sqrt(165) + Rational(32,9)*sqrt(Rational(5,33))*nu
          - Rational(32,9)*sqrt(Rational(5,33))*nu**2)
    + x**Rational(5,2)*(
        S*(Rational(-32,3)/sqrt(165) + Rational(32,1)*nu/sqrt(165))
        + delta*(Rational(-32,3)/sqrt(165) + Rational(64,3)*nu/sqrt(165))*Sigma
    )/(G*M**2)
    + x**Rational(7,2)*(
        S*(Rational(3856,39)/sqrt(165) - Rational(47024,117)*nu/sqrt(165)
           + Rational(3376,13)*nu**2/sqrt(165))
        + delta*(Rational(9904,117)/sqrt(165) - Rational(640,13)*sqrt(Rational(5,33))*nu
                 + Rational(13072,117)*nu**2/sqrt(165))*Sigma
    )/(G*M**2)
)

# ── Mode h[5,5] ───────────────────────────────────────────────────────────────
h55 = _pre(5) * (
    x**Rational(3,2)*delta*(Rational(625,96)*I/sqrt(66) - Rational(625,48)*I*nu/sqrt(66))
    + x**3*(
        S*delta*(Rational(-3125,72)*I/sqrt(66) + Rational(3125,36)*I*nu/sqrt(66))
        + (Rational(-3125,144)*I/sqrt(66) + Rational(15625,144)*I*nu/sqrt(66)
           - Rational(15625,144)*I*nu**2/sqrt(66))*Sigma
    )/(G*M**2)
    + x**Rational(7,2)*(
        (S**2*(delta*kappap*(Rational(3125,192)*I/sqrt(66) - Rational(3125,96)*I*nu/sqrt(66))
               + delta*(Rational(3125,96)*I/sqrt(66) - Rational(3125,48)*I*nu/sqrt(66)))
         + S*(Rational(3125,96)*I/sqrt(66) - Rational(3125,16)*I*nu/sqrt(66)
              + Rational(3125,12)*I*nu**2/sqrt(66)
              + delta*kappam*(Rational(-3125,192)*I/sqrt(66) + Rational(3125,96)*I*nu/sqrt(66))
              + kappap*(Rational(3125,192)*I/sqrt(66) - Rational(3125,32)*I*nu/sqrt(66)
                        + Rational(3125,24)*I*nu**2/sqrt(66))
             )*Sigma
         + (delta*kappap*(Rational(3125,384)*I/sqrt(66) - Rational(3125,96)*I*nu/sqrt(66)
                          + Rational(3125,96)*I*nu**2/sqrt(66))
            + kappam*(Rational(-3125,384)*I/sqrt(66) + Rational(3125,64)*I*nu/sqrt(66)
                      - Rational(3125,48)*I*nu**2/sqrt(66))
            + delta*(Rational(-3125,96)*I*nu/sqrt(66) + Rational(3125,48)*I*nu**2/sqrt(66))
           )*Sigma**2
        )/(G**2*M**4)
    )
)

# ── Mode h[6,1] ───────────────────────────────────────────────────────────────
# Note: prefactor uses G*M in numerator differently — here the Mathematica form
# is 8*sqrt(pi/5)*x^4*nu / (c^2 * exp(I*psi) * M * R)  [no G*M factor out front]
h61 = (8*sqrt(pi/5)*x**4*nu / (c**2 * exp(I*psi) * M * R)) * (
    S*delta*(I/2376/sqrt(26) - I/1188*nu/sqrt(26))
    + (I/2376/sqrt(26) - Rational(5,2376)*I*nu/sqrt(26) + Rational(5,2376)*I*nu**2/sqrt(26))*Sigma
)

# ── Mode h[6,2] ───────────────────────────────────────────────────────────────
h62 = _pre(2) * (
    x**2*(Rational(2,297)/sqrt(65) - Rational(2,297)*sqrt(Rational(5,13))*nu
          + Rational(2,297)*sqrt(Rational(5,13))*nu**2)
    + x**Rational(7,2)*(
        S*(Rational(4,693)/sqrt(65) - Rational(4,693)*sqrt(Rational(5,13))*nu
           + Rational(4,693)*sqrt(Rational(5,13))*nu**2)
        + delta*(Rational(68,2079)/sqrt(65) - Rational(272,2079)*nu/sqrt(65)
                 + Rational(68,693)*nu**2/sqrt(65))*Sigma
    )/(G*M**2)
)

# ── Mode h[6,3] ───────────────────────────────────────────────────────────────
h63 = (8*sqrt(pi/5)*x**4*nu / (c**2 * exp(3*I*psi) * M * R)) * (
    S*delta*(Rational(-81,176)*I/sqrt(65) + Rational(81,88)*I*nu/sqrt(65))
    + (Rational(-81,176)*I/sqrt(65) + Rational(81,176)*I*sqrt(Rational(5,13))*nu
       - Rational(81,176)*I*sqrt(Rational(5,13))*nu**2)*Sigma
)

# ── Mode h[6,4] ───────────────────────────────────────────────────────────────
h64 = _pre(4) * (
    x**2*(Rational(-128,495)*sqrt(Rational(2,39)) + Rational(128,99)*sqrt(Rational(2,39))*nu
          - Rational(128,99)*sqrt(Rational(2,39))*nu**2)
    + x**Rational(7,2)*(
        S*(Rational(256,385)*sqrt(Rational(2,39)) - Rational(256,77)*sqrt(Rational(2,39))*nu
           + Rational(256,77)*sqrt(Rational(2,39))*nu**2)
        + delta*(Rational(-256,693)*sqrt(Rational(2,39)) + Rational(1024,693)*sqrt(Rational(2,39))*nu
                 - Rational(256,231)*sqrt(Rational(2,39))*nu**2)*Sigma
    )/(G*M**2)
)

# ── Mode h[6,5] ───────────────────────────────────────────────────────────────
h65 = (8*sqrt(pi/5)*x**4*nu / (c**2 * exp(5*I*psi) * M * R)) * (
    S*delta*(Rational(3125,144)*I/sqrt(429) - Rational(3125,72)*I*nu/sqrt(429))
    + (Rational(3125,144)*I/sqrt(429) - Rational(15625,144)*I*nu/sqrt(429)
       + Rational(15625,144)*I*nu**2/sqrt(429))*Sigma
)

# ── Mode h[6,6] ───────────────────────────────────────────────────────────────
h66 = _pre(6) * (
    x**2*(Rational(54,5)/sqrt(143) - 54*nu/sqrt(143) + 54*nu**2/sqrt(143))
    + x**Rational(7,2)*(
        S*(Rational(-3132,35)/sqrt(143) + Rational(3132,7)*nu/sqrt(143)
           - Rational(3132,7)*nu**2/sqrt(143))
        + delta*(Rational(-324,7)/sqrt(143) + Rational(1296,7)*nu/sqrt(143)
                 - Rational(972,7)*nu**2/sqrt(143))*Sigma
    )/(G*M**2)
)

# ── Mode h[7,2] ───────────────────────────────────────────────────────────────
h72 = (8*sqrt(pi/5)*x**Rational(9,2)*nu / (c**2 * exp(2*I*psi) * M * R)) * (
    S*(Rational(4,3003)/sqrt(3) - Rational(20,3003)*nu/sqrt(3) + Rational(20,3003)*nu**2/sqrt(3))
    + delta*(Rational(4,3003)/sqrt(3) - Rational(16,3003)*nu/sqrt(3)
             + Rational(4,1001)*nu**2/sqrt(3))*Sigma
)

# ── Mode h[7,4] ───────────────────────────────────────────────────────────────
h74 = (8*sqrt(pi/5)*x**Rational(9,2)*nu / (c**2 * exp(4*I*psi) * M * R)) * (
    S*(Rational(-512,1365)*sqrt(Rational(2,33)) + Rational(512,273)*sqrt(Rational(2,33))*nu
       - Rational(512,273)*sqrt(Rational(2,33))*nu**2)
    + delta*(Rational(-512,1365)*sqrt(Rational(2,33)) + Rational(2048,1365)*sqrt(Rational(2,33))*nu
             - Rational(512,455)*sqrt(Rational(2,33))*nu**2)*Sigma
)

# ── Mode h[7,6] ───────────────────────────────────────────────────────────────
h76 = (8*sqrt(pi/5)*x**Rational(9,2)*nu / (c**2 * exp(6*I*psi) * M * R)) * (
    S*(Rational(324,35)*sqrt(Rational(3,143)) - Rational(324,7)*sqrt(Rational(3,143))*nu
       + Rational(324,7)*sqrt(Rational(3,143))*nu**2)
    + delta*(Rational(324,35)*sqrt(Rational(3,143)) - Rational(1296,35)*sqrt(Rational(3,143))*nu
             + Rational(972,35)*sqrt(Rational(3,143))*nu**2)*Sigma
)

# ── Collect all modes ─────────────────────────────────────────────────────────
modes = {
    (2, 1): h21,
    (2, 2): h22,
    (3, 1): h31,
    (3, 2): h32,
    (3, 3): h33,
    (4, 1): h41,
    (4, 2): h42,
    (4, 3): h43,
    (4, 4): h44,
    (5, 1): h51,
    (5, 2): h52,
    (5, 3): h53,
    (5, 4): h54,
    (5, 5): h55,
    (6, 1): h61,
    (6, 2): h62,
    (6, 3): h63,
    (6, 4): h64,
    (6, 5): h65,
    (6, 6): h66,
    (7, 2): h72,
    (7, 4): h74,
    (7, 6): h76,
}