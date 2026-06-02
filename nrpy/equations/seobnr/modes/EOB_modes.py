import sympy as sp

# ============================================================
# Symbols
# ============================================================

nu, delta = sp.symbols('nu delta')
x, e, ei, zeta = sp.symbols('x e ei zeta')
r, pr, phi, L = sp.symbols('r pr phi L')

chi1, chi2 = sp.symbols('chi1 chi2')
chiS, chiA = sp.symbols('chiS chiA')

kappa1, kappa2 = sp.symbols('kappa1 kappa2')
kappaS, kappaA = sp.symbols('kappaS kappaA')

alpha, beta = sp.symbols('alpha beta')
SO = sp.symbols('SO')
epsilon = sp.symbols('epsilon')

M, R = sp.symbols('M R')

lambdap, lambdam = sp.symbols('lambdap lambdam')  # included for completeness if needed later

x0 = sp.sqrt(sp.E) / 2
r0 = 1 / x0
x0p = sp.exp(sp.Rational(11, 18) - sp.Rational(2, 3) * sp.EulerGamma - sp.Rational(4, 3) * sp.log(2) + sp.Rational(2, 3) * sp.log(x0))
EulerGammaRedefined = sp.exp(-sp.Rational(11, 12) + sp.EulerGamma + 2 * sp.log(2))

# ============================================================
# H[l,m]
# Table is for l = 2..8 and m = 0..l
# ============================================================

H = {}

# ---------------- l = 2 ----------------

H[2, 0] = 0

H[2, 1] = -sp.I / 7560 * (
    -2520 * sp.sqrt(x) * delta
    - 90 * x**sp.Rational(3, 2) * delta * (-17 + 20 * nu)
    + 3780 * x * (chiA + delta * chiS)
    - 180 * x**2 * ((-7 + 205 * nu) * chiA + delta * (-7 + 33 * nu) * chiS)
    + 5 * x**sp.Rational(5, 2) * (
        252 * (kappaA * (-1 + 8 * nu) + 4 * (-3 + 10 * nu) * chiA * chiS)
        + delta * (
            -237 * nu**2
            + 252 * kappaS * (-1 + 6 * nu)
            + 4 * nu * (509 + 756 * chiA**2 + 252 * chiS**2)
            - 4 * (-43 + 378 * chiA**2 + 378 * chiS**2)
        )
    )
    + x**3 * (
        -5 * (
            378 * (-1 + 4 * nu) * chiA**3
            + 378 * kappaA * (-1 + 4 * nu) * chiS
            + chiA * (
                -1248 + 8351 * nu - 5460 * nu**2
                + 378 * kappaS * (-1 + 2 * nu)
                - 1134 * chiS**2 + 3024 * nu * chiS**2
            )
        )
        + delta * (
            -3580 * nu**2 * chiS
            + nu * (
                27468 * sp.I - 4032 * sp.I * alpha + 2688 * sp.I * beta
                - 35135 * chiS - 3780 * kappaS * chiS - 7560 * chiA**2 * chiS
            )
            + 30 * (
                63 * kappaA * chiA
                + chiS * (208 + 63 * kappaS + 189 * chiA**2 + 63 * chiS**2)
            )
        )
    )
)

H[2, 2] = (
    1
    + sp.Rational(1, 42) * x * (-107 + 55 * nu)
    - sp.Rational(4, 3) * x**sp.Rational(3, 2) * (delta * chiA + chiS - nu * chiS)
    + sp.Rational(2, 315) * x**sp.Rational(5, 2) * (
        660 * nu**2 * chiS
        - 400 * (delta * chiA + chiS)
        + nu * (-3276 * sp.I + 504 * sp.I * alpha - 336 * sp.I * beta - 280 * delta * chiA + 505 * chiS)
    )
    + x**2 * (
        -sp.Rational(2173, 1512)
        + kappaS
        - sp.Rational(1069, 216) * nu
        - 2 * kappaS * nu
        + sp.Rational(2047, 1512) * nu**2
        + chiA**2 - 4 * nu * chiA**2 + chiS**2
        + delta * (kappaA + 2 * chiA * chiS)
    )
    + x**3 * (
        sp.Rational(761273, 13200)
        - sp.Rational(278185, 33264) * nu
        + sp.Rational(41, 96) * sp.pi**2 * nu
        - sp.Rational(20261, 2772) * nu**2
        + sp.Rational(114635, 99792) * nu**3
        - sp.Rational(1, 21) * kappaS * (-12 + 45 * nu + 68 * nu**2)
        + sp.Rational(8, 63) * chiA**2
        + sp.Rational(73, 63) * nu * chiA**2
        - sp.Rational(136, 21) * nu**2 * chiA**2
        - sp.Rational(4, 3) * sp.I * chiS
        + sp.Rational(8, 3) * sp.I * nu * chiS
        + sp.Rational(8, 63) * chiS**2
        - sp.Rational(67, 9) * nu * chiS**2
        + sp.Rational(8, 3) * nu**2 * chiS**2
        - sp.Rational(1, 63) * delta * (
            9 * kappaA * (-4 + 7 * nu)
            + 4 * chiA * (21 * sp.I - 4 * chiS + 91 * nu * chiS)
        )
    )
    - sp.Rational(856, 105) * x**3 * sp.log(x0 / x)
)

# ---------------- l = 3 ----------------

H[3, 0] = -(sp.Rational(2, 5)) * sp.I * sp.sqrt(sp.Rational(6, 7)) * x**sp.Rational(5, 2) * nu

H[3, 1] = -(sp.I / (11880 * sp.sqrt(14))) * sp.sqrt(x) * (
    55 * x**sp.Rational(3, 2) * (
        9 * (4 - 11 * nu) * chiA
        + x * (-70 - 59 * nu + 931 * nu**2) * chiA
        - 9 * sp.sqrt(x) * (-5 + 4 * nu) * (kappaA + 2 * chiA * chiS)
    )
    + delta * (
        -990
        + 660 * x * (4 + nu)
        - 495 * x**sp.Rational(3, 2) * (-4 + 13 * nu) * chiS
        + 5 * x**2 * (
            -607 + 247 * nu**2 + 99 * kappaS * (5 + 6 * nu)
            + 495 * chiA**2 + 4 * nu * (68 + 297 * chiA**2) + 495 * chiS**2
        )
        + 11 * x**sp.Rational(5, 2) * (
            -350 * chiS + 225 * nu**2 * chiS
            + 3 * nu * (971 * sp.I - 48 * sp.I * alpha + 32 * sp.I * beta + 165 * chiS)
        )
    )
)

H[3, 2] = -(1 / (2376 * sp.sqrt(35))) * x * (
    3960 * (-1 + 3 * nu)
    + 44 * x * (193 - 725 * nu + 365 * nu**2)
    - 15840 * sp.sqrt(x) * nu * chiS
    + 264 * x**sp.Rational(3, 2) * (
        130 * nu**2 * chiS
        + 40 * (delta * chiA + chiS)
        + nu * (-63 * sp.I - 215 * delta * chiA + 15 * chiS)
    )
    + x**2 * (
        1451 + 17387 * nu - 100026 * nu**2 + 16023 * nu**3
        - 7920 * kappaS * (1 - 4 * nu + 6 * nu**2)
        - 7920 * chiA**2 + 63360 * nu * chiA**2 - 95040 * nu**2 * chiA**2
        - 15840 * sp.I * chiS + 31680 * sp.I * nu * chiS
        - 7920 * chiS**2 + 52800 * nu * chiS**2 - 42240 * nu**2 * chiS**2
        + 2640 * delta * (
            kappaA * (-3 + 6 * nu)
            + 2 * chiA * (-3 * sp.I - 3 * chiS + 16 * nu * chiS)
        )
    )
)

H[3, 3] = -(sp.I / (792 * sp.sqrt(210))) * sp.sqrt(x) * (
    297 * x**sp.Rational(3, 2) * (
        15 * (-4 + 19 * nu) * chiA
        + x * (10 - 279 * nu + 407 * nu**2) * chiA
        - 45 * sp.sqrt(x) * (-1 + 4 * nu) * (kappaA + 2 * chiA * chiS)
    )
    + delta * (
        8910
        + 17820 * x * (-2 + nu)
        + 4455 * x**sp.Rational(3, 2) * (-4 + 5 * nu) * chiS
        + 11 * x**sp.Rational(5, 2) * (
            sp.I * nu * (-23107 + 3888 * alpha - 2592 * beta + 27 * sp.I * chiS)
            + 270 * chiS + 6507 * nu**2 * chiS
        )
        + 27 * x**2 * (
            887 * nu**2
            - 495 * kappaS * (-1 + 2 * nu)
            - 4 * nu * (919 + 495 * chiA**2)
            + 9 * (41 + 55 * chiA**2 + 55 * chiS**2)
        )
    )
)

# ---------------- l = 4 ----------------

H[4, 0] = 0

H[4, 1] = -(sp.I / (55440 * sp.sqrt(10))) * x**sp.Rational(3, 2) * (
    10 * sp.sqrt(x) * (165 * nu + x * (220 - 2247 * nu + 2891 * nu**2)) * chiA
    + delta * (
        660 * (-1 + 2 * nu)
        + 5 * x * (404 - 1011 * nu + 332 * nu**2)
        - 1650 * sp.sqrt(x) * nu * chiS
        + x**sp.Rational(3, 2) * (
            -33726 * sp.I * nu + 2200 * chiS + 470 * nu * chiS + 6130 * nu**2 * chiS
        )
    )
)

H[4, 2] = (1 / (22702680 * sp.sqrt(5))) * x * (
    -1801800 * (-1 + 3 * nu)
    - 5460 * x * (1311 - 4025 * nu + 285 * nu**2)
    - 120120 * x**sp.Rational(3, 2) * (
        312 * nu**2 * chiS
        + 40 * (delta * chiA + chiS)
        - nu * (63 * sp.I + 84 * delta * chiA + 236 * chiS)
    )
    + x**2 * (
        9342351 - 38225313 * nu + 28031710 * nu**2 + 2707215 * nu**3
        + 3603600 * kappaS * (1 - 2 * nu + 6 * nu**2)
        + 3603600 * chiA**2 - 14414400 * nu * chiA**2 + 43243200 * nu**2 * chiA**2
        + 3603600 * chiS**2 + 3603600 * delta * (kappaA + 2 * chiA * chiS)
    )
)

H[4, 3] = (sp.I / (7920 * sp.sqrt(70))) * (
    17820 * x**sp.Rational(3, 2) * delta * (-1 + 2 * nu)
    + 135 * x**sp.Rational(5, 2) * delta * (468 - 1267 * nu + 524 * nu**2)
    + 44550 * x**2 * nu * (chiA - delta * chiS)
    + 2 * x**3 * (
        135 * (220 - 2403 * nu + 3359 * nu**2) * chiA
        + delta * (-65263 * sp.I * nu + 29700 * chiS + 27405 * nu * chiS + 61695 * nu**2 * chiS)
    )
)

H[4, 4] = (1 / (405405 * sp.sqrt(35))) * x * (
    1801800 * (-1 + 3 * nu)
    + 5460 * x * (1779 - 6365 * nu + 2625 * nu**2)
    + 30030 * x**sp.Rational(3, 2) * (
        672 * nu**2 * chiS
        + 160 * (delta * chiA + chiS)
        - nu * (279 * sp.I + 624 * delta * chiA + 656 * chiS)
    )
    + x**2 * (
        -9618039 + 68551497 * nu - 113096830 * nu**2 + 23740185 * nu**3
        - 3603600 * kappaS * (1 - 5 * nu + 6 * nu**2)
        - 3603600 * chiA**2 + 25225200 * nu * chiA**2 - 43243200 * nu**2 * chiA**2
        - 3603600 * chiS**2 + 10810800 * nu * chiS**2
        + 3603600 * delta * (-1 + 3 * nu) * (kappaA + 2 * chiA * chiS)
    )
)

# ---------------- l = 5 ----------------

H[5, 0] = 0

H[5, 1] = -(sp.I / (56160 * sp.sqrt(385))) * x**sp.Rational(3, 2) * (
    130 * x**sp.Rational(3, 2) * (5 - 23 * nu + 19 * nu**2) * chiA
    + delta * (
        195 * (-1 + 2 * nu)
        + 5 * x * (179 - 352 * nu + 4 * nu**2)
        + 13 * x**sp.Rational(3, 2) * (
            50 * chiS + 270 * nu**2 * chiS - 9 * nu * (101 * sp.I + 30 * chiS)
        )
    )
)

H[5, 2] = (
    x**2 * (
        910 * (1 - 5 * nu + 5 * nu**2)
        + x * (-3911 + 21553 * nu - 28910 * nu**2 + 8085 * nu**3)
        - 2730 * sp.sqrt(x) * nu * (delta * chiA + (-1 + 2 * nu) * chiS)
    )
) / (12285 * sp.sqrt(55))

H[5, 3] = (sp.I / (131040 * sp.sqrt(330))) * x**sp.Rational(3, 2) * (
    110565 * delta * (-1 + 2 * nu)
    + 2835 * x * delta * (207 - 464 * nu + 88 * nu**2)
    + 13 * x**sp.Rational(3, 2) * (
        5670 * (5 - 27 * nu + 31 * nu**2) * chiA
        + delta * (
            28350 * chiS + 130410 * nu**2 * chiS
            - nu * (63409 * sp.I + 130410 * chiS)
        )
    )
)

H[5, 4] = -(16 / (4095 * sp.sqrt(165))) * x**2 * (
    910 * (1 - 5 * nu + 5 * nu**2)
    + x * (-4451 + 25333 * nu - 36470 * nu**2 + 11865 * nu**3)
    - 2730 * sp.sqrt(x) * nu * (delta * chiA + (-1 + 2 * nu) * chiS)
)

H[5, 5] = -(sp.I / (131040 * sp.sqrt(66))) * x**sp.Rational(3, 2) * (
    2843750 * x**sp.Rational(3, 2) * (1 - 7 * nu + 11 * nu**2) * chiA
    + delta * (
        853125 * (-1 + 2 * nu)
        + 21875 * x * (263 - 688 * nu + 256 * nu**2)
        + 13 * x**sp.Rational(3, 2) * (
            -528231 * sp.I * nu + 218750 * chiS - 656250 * nu * chiS + 656250 * nu**2 * chiS
        )
    )
)

# ---------------- l = 6 ----------------

H[6, 0] = 0

H[6, 1] = -(sp.I / (16632 * sp.sqrt(26))) * (
    7 * x**3 * (1 - 3 * nu) * nu * chiA
    + x**sp.Rational(5, 2) * delta * (-1 + nu) * (2 + nu * (-6 + 7 * sp.sqrt(x) * chiS))
)

H[6, 2] = (
    x**2 * (
        14 * (1 - 5 * nu + 5 * nu**2)
        + x * (-81 + 413 * nu - 448 * nu**2 + 49 * nu**3)
    )
) / (2079 * sp.sqrt(65))

H[6, 3] = (81 * sp.I / (1232 * sp.sqrt(65))) * (
    7 * x**3 * (1 - 3 * nu) * nu * chiA
    + x**sp.Rational(5, 2) * delta * (-1 + nu) * (2 + nu * (-6 + 7 * sp.sqrt(x) * chiS))
)

H[6, 4] = -(64 * sp.sqrt(sp.Rational(2, 39)) / 3465) * x**2 * (
    14 * (1 - 5 * nu + 5 * nu**2)
    + x * (-93 + 497 * nu - 616 * nu**2 + 133 * nu**3)
)

H[6, 5] = -(3125 * sp.I / (1008 * sp.sqrt(429))) * (
    7 * x**3 * (1 - 3 * nu) * nu * chiA
    + x**sp.Rational(5, 2) * delta * (-1 + nu) * (2 + nu * (-6 + 7 * sp.sqrt(x) * chiS))
)

H[6, 6] = (27 / (35 * sp.sqrt(143))) * x**2 * (
    14 * (1 - 5 * nu + 5 * nu**2)
    + x * (-113 + 637 * nu - 896 * nu**2 + 273 * nu**3)
)

# ---------------- l = 7 ----------------

H[7, 0] = 0

H[7, 1] = (sp.I / (864864 * sp.sqrt(2))) * x**sp.Rational(5, 2) * delta * (1 - 4 * nu + 3 * nu**2)

H[7, 2] = -x**3 * (-1 + 7 * nu - 14 * nu**2 + 7 * nu**3) / (3003 * sp.sqrt(3))

H[7, 3] = -(243 * sp.I * sp.sqrt(sp.Rational(3, 2)) / 160160) * x**sp.Rational(5, 2) * delta * (1 - 4 * nu + 3 * nu**2)

H[7, 4] = (128 * sp.sqrt(sp.Rational(2, 33)) / 1365) * x**3 * (-1 + 7 * nu - 14 * nu**2 + 7 * nu**3)

H[7, 5] = (15625 * sp.I / (26208 * sp.sqrt(66))) * x**sp.Rational(5, 2) * delta * (1 - 4 * nu + 3 * nu**2)

H[7, 6] = -(sp.Rational(81, 35)) * sp.sqrt(sp.Rational(3, 143)) * x**3 * (-1 + 7 * nu - 14 * nu**2 + 7 * nu**3)

H[7, 7] = -(16807 * sp.I * sp.sqrt(sp.Rational(7, 858)) / 1440) * x**sp.Rational(5, 2) * delta * (1 - 4 * nu + 3 * nu**2)

# ---------------- l = 8 ----------------

H[8, 0] = 0
H[8, 1] = 0

H[8, 2] = -x**3 * (-1 + 7 * nu - 14 * nu**2 + 7 * nu**3) / (9009 * sp.sqrt(85))

H[8, 3] = 0

H[8, 4] = (128 * sp.sqrt(sp.Rational(2, 187)) / 4095) * x**3 * (-1 + 7 * nu - 14 * nu**2 + 7 * nu**3)

H[8, 5] = 0

H[8, 6] = -(sp.Rational(243, 35)) * sp.sqrt(sp.Rational(3, 17017)) * x**3 * (-1 + 7 * nu - 14 * nu**2 + 7 * nu**3)

H[8, 7] = 0

H[8, 8] = (sp.Rational(16384, 63)) * sp.sqrt(sp.Rational(2, 85085)) * x**3 * (-1 + 7 * nu - 14 * nu**2 + 7 * nu**3)