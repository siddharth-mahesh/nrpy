import sympy as sp

# Symbols
nu, delta, v = sp.symbols('nu delta v')
chiA, chiS = sp.symbols('chiA chiS')
kappaA, kappaS = sp.symbols('kappaA kappaS')
lambdaA, lambdaS = sp.symbols('lambdaA lambdaS')
omegaE = sp.symbols('omegaE')

# Containers
f = {}
rho = {}
delta_phase = {}

# --------------------------------------------------
# f_lm
# --------------------------------------------------

f[2, 1] = (
    1
    + v**2 * (-177 + 46*nu) / 84
    + v * ((-3*chiA)/(2*delta) - 3*chiS/2)
    + v**3 * (((427 + 131*nu)*chiA)/(84*delta) + ((427 + 79*nu)*chiS)/84)
    + v**4 * (
        -kappaA/(2*delta)
        + kappaS*(-sp.Rational(1, 2) - nu)
        + (-140 - 538*nu + 85*nu**2)/252
        + (-3 - 2*nu)*chiA**2
        + (3*(-4 + 7*nu)*chiA*chiS)/(2*delta)
        + ((-6 + nu)*chiS**2)/2
    )
    + v**6 * (
        kappaS*(2 + 20*nu + 15*nu**2)/42
        + kappaA*(4 + 32*nu + 473*nu**2)/(84*delta)
        + ((16652 - 9287*nu + 720*nu**2)*chiA**2)/1008
        + ((16652 - 39264*nu + 9487*nu**2)*chiA*chiS)/(504*delta)
        + ((16652 - 2633*nu + 1946*nu**2)*chiS**2)/1008
    )
    + v**5 * (
        -((5103 - 8797*nu + 6327*nu**2)*chiA)/(1008*delta)
        + ((3 - 12*nu)*chiA**3)/(4*delta)
        + (((-5103 + 1709*nu + 613*nu**2)/1008) + (sp.Rational(9, 4) - 3*nu)*chiA**2)*chiS
        + ((9 - 24*nu)*chiA*chiS**2)/(4*delta)
        + 3*chiS**3/4
        + kappaA*(3*chiA/4 + ((3 - 12*nu)*chiS)/(4*delta))
        + kappaS*(((3 - 6*nu)*chiA)/(4*delta) + (sp.Rational(3, 4) - 3*nu/2)*chiS)
    )
)

f[2, 2] = (
    1
    + v**2 * (-86 + 55*nu) / 42
    + v**3 * ((-4*delta*chiA)/3 + (4*(-1 + nu)*chiS)/3)
    + v**5 * (
        (-2*delta*(59 + 56*nu)*chiA)/63
        + (2*(-59 + 101*nu + 132*nu**2)*chiS)/63
    )
    + v**4 * (
        delta*kappaA
        + kappaS*(1 - 2*nu)
        + (-4288 - 6745*nu + 2047*nu**2)/1512
        + (1 - 4*nu)*chiA**2
        + 2*delta*chiA*chiS
        + chiS**2
    )
    + v**6 * (
        delta*kappaA*(sp.Rational(4, 7) - nu)
        + kappaS*(12 - 45*nu - 68*nu**2)/21
        + ((8 + 73*nu - 408*nu**2)*chiA**2)/63
        + (4*delta*(4 - 91*nu)*chiA*chiS)/63
        + (sp.Rational(8, 63) - 67*nu/9 + 8*nu**2/3)*chiS**2
    )
    + v**7 * (
        2*delta*lambdaA*(-1 + nu)
        + lambdaS*(-2 + 6*nu)
        + (delta*(6248 + 19771*nu + 1416*nu**2)*chiA)/756
        + (((6248 + 15155*nu - 7762*nu**2 + 3318*nu**3)/756) + (8*(1 - 4*nu)*nu*chiA**2)/3)*chiS
        + (16*delta*nu*chiA*chiS**2)/3
        + (8*nu*chiS**3)/3
        + kappaA*((2 - 2*nu)*chiA + (2*delta*(3 - 5*nu)*chiS)/3)
        + kappaS*(2*delta*(1 + nu)*chiA - (2*(-3 + 11*nu + 8*nu**2)*chiS)/3)
    )
)

f[3, 1] = (
    1
    + v**2 * (-13 - 4*nu) / 6
    + v**3 * (((-4 + 11*nu)*chiA)/(2*delta) + (-2 + 13*nu/2)*chiS)
    + v**5 * (((152 + 25*nu - 1862*nu**2)*chiA)/(36*delta) + (sp.Rational(38, 9) - 35*nu/12 - 5*nu**2/2)*chiS)
    + v**4 * (
        kappaS*(-sp.Rational(5, 2) - 3*nu)
        + (kappaA*(-5 + 4*nu))/(2*delta)
        + (-sp.Rational(5, 2) - 6*nu)*chiA**2
        + ((-5 + 4*nu)*chiA*chiS)/delta
        - 5*chiS**2/2
    )
    + v**6 * (
        kappaS*(43 - 8*nu + 88*nu**2)/12
        + kappaA*(43 - 94*nu + 304*nu**2)/(12*delta)
        + ((43 + 50*nu + 176*nu**2)*chiA**2)/12
        + ((43 - 166*nu + 904*nu**2)*chiA*chiS)/(6*delta)
        + (sp.Rational(43, 12) - 35*nu/2 + 22*nu**2)*chiS**2
    )
)

f[3, 2] = (
    1
    + v**2 * (328 - 1115*nu + 320*nu**2) / (90*(-1 + 3*nu))
    + (4*v*nu*chiS)/(1 - 3*nu)
    + v**3 * (
        (delta*(2 + 13*nu)*chiA)/(3 - 9*nu)
        + ((-2 + 56*nu + 13*nu**2)*chiS)/(-3 + 9*nu)
    )
    + v**5 * (
        (delta*(101024 - 13441*nu + 58177*nu**2)*chiA)/(11880*(-1 + 3*nu))
        + ((101024 - 517991*nu + 225141*nu**2 + 91342*nu**3)*chiS)/(11880*(-1 + 3*nu))
    )
    + v**4 * (
        (delta*kappaA*(-1 + nu))/(-1 + 3*nu)
        + (kappaS*(1 - 3*nu + 6*nu**2))/(1 - 3*nu)
        + (-20496 + 117215*nu - 253768*nu**2 + 39544*nu**3)/(11880*(-1 + 3*nu))
        + ((1 - 9*nu + 12*nu**2)*chiA**2)/(1 - 3*nu)
        + (2*delta*(-1 + nu)*chiA*chiS)/(-1 + 3*nu)
        + ((1 + 3*nu + 4*nu**2)*chiS**2)/(1 - 3*nu)
    )
)

f[3, 3] = (
    1
    + v**2 * (-sp.Rational(7, 2) + 2*nu)
    + v**3 * (((-4 + 19*nu)*chiA)/(2*delta) + (-2 + 5*nu/2)*chiS)
    + v**5 * (((40 - 593*nu + 814*nu**2)*chiA)/(60*delta) + ((40 + 33*nu + 482*nu**2)*chiS)/60)
    + v**4 * (
        (kappaA*(3 - 12*nu))/(2*delta)
        + kappaS*(sp.Rational(3, 2) - 3*nu)
        + (sp.Rational(3, 2) - 6*nu)*chiA**2
        + ((3 - 12*nu)*chiA*chiS)/delta
        + 3*chiS**2/2
    )
    + v**6 * (
        kappaS*(-sp.Rational(7, 4) + 2*nu - 6*nu**2)
        + (kappaA*(-7 + 22*nu + 16*nu**2))/(4*delta)
        + ((-7 + 22*nu - 48*nu**2)*chiA**2)/4
        - ((7 + 2*nu - 88*nu**2)*chiA*chiS)/(2*delta)
        + (-sp.Rational(7, 4) - 27*nu/2 + 6*nu**2)*chiS**2
    )
)

f[4, 1] = (
    1
    + v**2 * (602 - 1385*nu + 288*nu**2) / (132*(-1 + 2*nu))
    + v * ((-5*nu*chiA)/(2*delta - 4*delta*nu) + (5*nu*chiS)/(2 - 4*nu))
    + v**3 * (
        (nu*(-2349 + 2207*nu)*chiA)/(132*delta*(-1 + 2*nu))
        + (nu*(1689 + 841*nu)*chiS)/(132*(-1 + 2*nu))
    )
    + v**4 * (
        kappaS*(sp.Rational(3, 2) - 3*nu)
        + (kappaA*(3 - 18*nu - 4*nu**2))/(2*delta - 4*delta*nu)
        + ((9 - 74*nu + 72*nu**2)*chiA**2)/(6 - 12*nu)
        + ((18 - 108*nu + 41*nu**2)*chiA*chiS)/(6*delta - 12*delta*nu)
        + ((9 + 2*nu + 35*nu**2)*chiS**2)/(6 - 12*nu)
    )
)

f[4, 2] = (
    1
    + v**2 * (1146 - 3530*nu + 285*nu**2) / (330*(-1 + 3*nu))
    + v**3 * (
        (delta*(40 - 84*nu)*chiA)/(-15 + 45*nu)
        + (4*(10 - 59*nu + 78*nu**2)*chiS)/(-15 + 45*nu)
    )
    + v**5 * (
        (delta*(-368 + 81*nu + 1986*nu**2)*chiA)/(55*(-1 + 3*nu))
        + ((1104 - 5481*nu + 6142*nu**2 - 1134*nu**3)*chiS)/(165 - 495*nu)
    )
    + v**4 * (
        (2*delta*kappaA)/(1 - 3*nu)
        + (kappaS*(2 - 4*nu + 12*nu**2))/(1 - 3*nu)
        + ((2 - 8*nu + 24*nu**2)*chiA**2)/(1 - 3*nu)
        + (4*delta*chiA*chiS)/(1 - 3*nu)
        + (2*chiS**2)/(1 - 3*nu)
    )
)

f[4, 3] = (
    1
    + v**2 * (222 - 547*nu + 160*nu**2) / (-44 + 88*nu)
    + v * ((-5*nu*chiA)/(2*delta - 4*delta*nu) + (5*nu*chiS)/(2 - 4*nu))
    + v**3 * (
        (nu*(-2661 + 3143*nu)*chiA)/(132*delta*(-1 + 2*nu))
        + (23*nu*(87 + 23*nu)*chiS)/(132*(-1 + 2*nu))
    )
    + v**4 * (
        kappaS*(sp.Rational(3, 2) - 3*nu)
        + (kappaA*(3 - 18*nu + 12*nu**2))/(2*delta - 4*delta*nu)
        + ((9 - 74*nu + 72*nu**2)*chiA**2)/(6 - 12*nu)
        + ((18 - 108*nu + 137*nu**2)*chiA*chiS)/(6*delta - 12*delta*nu)
        + ((9 + 2*nu + 35*nu**2)*chiS**2)/(6 - 12*nu)
    )
)

f[4, 4] = (
    1
    + v**2 * (1614 - 5870*nu + 2625*nu**2) / (330*(-1 + 3*nu))
    + v**3 * (
        (4*delta*(-10 + 39*nu)*chiA)/(15 - 45*nu)
        + (4*(10 - 41*nu + 42*nu**2)*chiS)/(-15 + 45*nu)
    )
    + v**5 * (
        -(delta*(262 - 1845*nu + 1038*nu**2)*chiA)/(55*(-1 + 3*nu))
        + ((786 - 3501*nu + 5326*nu**2 - 6630*nu**3)*chiS)/(165 - 495*nu)
    )
    + v**4 * (
        2*delta*kappaA
        + kappaS*(2 - 4*nu)
        + (2 - 8*nu)*chiA**2
        + 4*delta*chiA*chiS
        + 2*chiS**2
    )
)

f[5, 1] = (
    1
    + v**3 * (
        -((10 - 46*nu + 38*nu**2)*chiA)/(3*delta - 6*delta*nu)
        + ((10 - 54*nu + 54*nu**2)*chiS)/(-3 + 6*nu)
    )
    + v**4 * (
        kappaS*(sp.Rational(5, 2) - 5*nu)
        + (kappaA*(5 - 30*nu - 8*nu**2))/(2*delta - 4*delta*nu)
        + (sp.Rational(5, 2) - 10*nu)*chiA**2
        + ((5 - 30*nu - 8*nu**2)*chiA*chiS)/(delta - 2*delta*nu)
        + 5*chiS**2/2
    )
)

f[5, 2] = (
    1
    + v**2 * (-15828 + 84679*nu - 104930*nu**2 + 21980*nu**3) / (2730*(1 - 5*nu + 5*nu**2))
    + v * ((-3*delta*nu*chiA)/(1 - 5*nu + 5*nu**2) + (3*(1 - 2*nu)*nu*chiS)/(1 - 5*nu + 5*nu**2))
    + v**3 * (
        -(delta*(52 - 2216*nu + 1403*nu**2)*chiA)/(78*(1 - 5*nu + 5*nu**2))
        + ((-52 - 1202*nu + 2325*nu**2 + 1522*nu**3)*chiS)/(78*(1 - 5*nu + 5*nu**2))
    )
)

f[5, 3] = (
    1
    + v**3 * (
        -((10 - 54*nu + 62*nu**2)*chiA)/(3*delta - 6*delta*nu)
        + ((10 - 46*nu + 46*nu**2)*chiS)/(-3 + 6*nu)
    )
    + v**4 * (
        kappaS*(sp.Rational(5, 2) - 5*nu)
        + (kappaA*(5 - 30*nu + 8*nu**2))/(2*delta - 4*delta*nu)
        + (sp.Rational(5, 2) - 10*nu)*chiA**2
        + ((5 - 30*nu + 8*nu**2)*chiA*chiS)/(delta - 2*delta*nu)
        + 5*chiS**2/2
    )
)

f[5, 4] = (
    1
    + v**2 * (-17448 + 96019*nu - 127610*nu**2 + 33320*nu**3) / (2730*(1 - 5*nu + 5*nu**2))
    + v * ((-3*delta*nu*chiA)/(1 - 5*nu + 5*nu**2) + (3*(1 - 2*nu)*nu*chiS)/(1 - 5*nu + 5*nu**2))
    + v**3 * (
        -(delta*(52 - 2468*nu + 1907*nu**2)*chiA)/(78*(1 - 5*nu + 5*nu**2))
        + ((-52 - 1454*nu + 3333*nu**2 + 1018*nu**3)*chiS)/(78*(1 - 5*nu + 5*nu**2))
    )
)

f[5, 5] = (
    1
    + v**3 * (
        -((10 - 70*nu + 110*nu**2)*chiA)/(3*delta - 6*delta*nu)
        + (10*(1 - 3*nu + 3*nu**2)*chiS)/(-3 + 6*nu)
    )
    + v**4 * (
        (kappaA*(5 - 20*nu))/(2*delta)
        + kappaS*(sp.Rational(5, 2) - 5*nu)
        + (sp.Rational(5, 2) - 10*nu)*chiA**2
        + ((5 - 20*nu)*chiA*chiS)/delta
        + 5*chiS**2/2
    )
)

f[6, 1] = 1 + v*((-7*nu*chiA)/(2*delta - 2*delta*nu) + (7*nu*chiS)/(2 - 6*nu))

f[6, 2] = (
    1
    + v**3 * (
        (-2*delta*(14 - 53*nu + 36*nu**2)*chiA)/(7*(1 - 5*nu + 5*nu**2))
        + (2*(-14 + 115*nu - 278*nu**2 + 174*nu**3)*chiS)/(7*(1 - 5*nu + 5*nu**2))
    )
)

f[6, 3] = 1 + v*((-7*nu*chiA)/(2*delta - 2*delta*nu) + (7*nu*chiS)/(2 - 6*nu))

f[6, 4] = (
    1
    + v**3 * (
        (-2*delta*(14 - 65*nu + 60*nu**2)*chiA)/(7*(1 - 5*nu + 5*nu**2))
        + (2*(-14 + 103*nu - 230*nu**2 + 150*nu**3)*chiS)/(7*(1 - 5*nu + 5*nu**2))
    )
)

f[6, 5] = 1 + v*((-7*nu*chiA)/(2*delta - 2*delta*nu) + (7*nu*chiS)/(2 - 6*nu))

f[6, 6] = (
    1
    + v**3 * (
        (-2*delta*(14 - 85*nu + 100*nu**2)*chiA)/(7*(1 - 5*nu + 5*nu**2))
        + (2*(-14 + 83*nu - 150*nu**2 + 110*nu**3)*chiS)/(7*(1 - 5*nu + 5*nu**2))
    )
)

f[7, 2] = (
    1
    + v * (
        (4*delta*(1 - 2*nu)*nu*chiA)/(-1 + 7*nu - 14*nu**2 + 7*nu**3)
        - (4*nu*(1 - 4*nu + 2*nu**2)*chiS)/(-1 + 7*nu - 14*nu**2 + 7*nu**3)
    )
)

f[7, 4] = f[7, 2]
f[7, 6] = f[7, 2]

# --------------------------------------------------
# rho_lm
# --------------------------------------------------

rho[2, 1] = (
    1
    + v*((-3*chiA)/(4*delta) - 3*chiS/4)
    + v**2*(-sp.Rational(59, 56) + 23*nu/84 + (9*chiA**2)/(32*(-1 + 4*nu)) - (9*chiA*chiS)/(16*delta) - 9*chiS**2/32)
    + v**4*(
        -kappaA/(4*delta)
        + (kappaS*(-1 - 2*nu))/4
        + (-47009 - 43972*nu + 7404*nu**2)/56448
        + ((865 - 10422*nu - 7168*nu**2)*chiA**2)/(1792*(-1 + 4*nu))
        + ((-865 + 5958*nu)*chiA*chiS)/(896*delta)
        + ((-865 + 1494*nu)*chiS**2)/1792
    )
    + v**6*(
        (kappaS*(-161 - 148*nu + 212*nu**2))/672
        + (kappaA*(-161 + 174*nu + 1892*nu**2))/(672*delta)
        + ((-9032393 + 54330988*nu - 33371924*nu**2 + 4558848*nu**3)*chiA**2)/(1806336*(-1 + 4*nu))
        + ((9032393 - 28363212*nu + 4226836*nu**2)*chiA*chiS)/(903168*delta)
        + ((9032393 - 2395436*nu + 1280276*nu**2)*chiS**2)/1806336
    )
    + v**3*(
        ((1177 + 662*nu)*chiA)/(672*delta)
        - (27*chiA**3)/(128*delta - 512*delta*nu)
        + (((1177 + 454*nu)/672) + (81*chiA**2)/(128*(-1 + 4*nu)))*chiS
        - (81*chiA*chiS**2)/(128*delta)
        - 27*chiS**3/128
    )
    + v**5*(
        -((295905 - 979412*nu + 747316*nu**2)*chiA)/(225792*delta)
        - (3*(677 + 4054*nu + 21504*nu**2)*chiA**3)/(7168*delta*(-1 + 4*nu))
        + (((-295905 + 111924*nu + 49100*nu**2)/225792) - (9*(677 + 3398*nu + 7168*nu**2)*chiA**2)/(7168*(-1 + 4*nu)))*chiS
        + (9*(677 + 2742*nu)*chiA*chiS**2)/(7168*delta)
        + (3*(677 + 2086*nu)*chiS**3)/7168
        + kappaA*((sp.Rational(3, 8) - sp.Rational(3, 1)/(16 - 64*nu))*chiA + ((3 - 24*nu)*chiS)/(16*delta))
        + kappaS*(((3 - 18*nu)*chiA)/(16*delta) + (sp.Rational(3, 16) - 9*nu/8)*chiS)
    )
)

rho[2, 2] = (
    1
    + v**2*(-86 + 55*nu)/84
    + v**3*((-2*delta*chiA)/3 + (2*(-1 + nu)*chiS)/3)
    + v**5*(-(delta*(68 + 19*nu)*chiA)/42 + ((-204 + 343*nu + 209*nu**2)*chiS)/126)
    + v**4*((delta*kappaA)/2 + kappaS*(sp.Rational(1, 2) - nu) + (-82220 - 66050*nu + 19583*nu**2)/42336 + (sp.Rational(1, 2) - 2*nu)*chiA**2 + delta*chiA*chiS + chiS**2/2)
    + v**6*((delta*kappaA*(134 - 139*nu))/168 + (kappaS*(134 - 407*nu - 162*nu**2))/168 + ((178 - 457*nu - 972*nu**2)*chiA**2)/504 + (delta*(178 - 781*nu)*chiA*chiS)/252 + ((178 - 1817*nu + 560*nu**2)*chiS**2)/504)
    + v**7*(delta*lambdaA*(-1 + nu) + lambdaS*(-1 + 3*nu) + (delta*(74932 + 802240*nu + 97865*nu**2)*chiA)/63504 + ((delta - 4*delta*nu)*chiA**3)/3 + (((74932 + 896988*nu - 245717*nu**2 + 50803*nu**3)/63504) + (1 - 3*nu - 4*nu**2)*chiA**2)*chiS + (delta + 2*delta*nu)*chiA*chiS**2 + (sp.Rational(1, 3) + nu)*chiS**3 + kappaA*(((4 - 7*nu)*chiA)/3 + delta*(sp.Rational(4, 3) - 2*nu)*chiS) + kappaS*((delta*(4 + nu)*chiA)/3 - (2*(-2 + 7*nu + 3*nu**2)*chiS)/3))
)

rho[3, 1] = (
    1
    + v**2*(-13 - 4*nu)/18
    + v**3*(((-4 + 11*nu)*chiA)/(6*delta) + ((-4 + 13*nu)*chiS)/6)
    + v**5*(((48 + 279*nu - 1774*nu**2)*chiA)/(108*delta) + (sp.Rational(4, 9) + 67*nu/36 + 7*nu**2/54)*chiS)
    + v**4*(kappaS*(-sp.Rational(5, 6) - nu) + (kappaA*(-5 + 4*nu))/(6*delta) + (-sp.Rational(5, 6) - 2*nu)*chiA**2 + ((-5 + 4*nu)*chiA*chiS)/(3*delta) - 5*chiS**2/6)
    + v**6*(-(kappaA*(1 + 218*nu - 944*nu**2))/(108*delta) + kappaS*(-sp.Rational(1, 108) - 55*nu/27 + 2*nu**2) + ((49 - 66*nu - 877*nu**2 + 1728*nu**3)*chiA**2)/(108*(-1 + 4*nu)) - ((49 + 146*nu - 2315*nu**2)*chiA*chiS)/(54*delta) + ((-49 - 358*nu + 285*nu**2)*chiS**2)/108)
)

rho[3, 2] = (
    1
    + (4*v*nu*chiS)/(3 - 9*nu)
    + v**2*((328 - 1115*nu + 320*nu**2)/(270*(-1 + 3*nu)) - (16*nu**2*chiS**2)/(9*(1 - 3*nu)**2))
    + v**4*((delta*kappaA*(-1 + nu))/(-3 + 9*nu) + (kappaS*(1 - 3*nu + 6*nu**2))/(3 - 9*nu) + (-1444528 + 8050045*nu - 4725605*nu**2 - 20338960*nu**3 + 3085640*nu**4)/(1603800*(1 - 3*nu)**2) + ((1 - 9*nu + 12*nu**2)*chiA**2)/(3 - 9*nu) - (2*delta*(-9 + 44*nu + 25*nu**2)*chiA*chiS)/(27*(1 - 3*nu)**2) + ((-81 + 387*nu - 1435*nu**2 + 1997*nu**3 + 2452*nu**4)*chiS**2)/(243*(-1 + 3*nu)**3))
    + v**3*((delta*(2 + 13*nu)*chiA)/(9 - 27*nu) + ((90 - 1478*nu + 2515*nu**2 + 3035*nu**3)*chiS)/(405*(1 - 3*nu)**2) - (320*nu**3*chiS**3)/(81*(-1 + 3*nu)**3))
    + v**5*((delta*(-245344 + 1128531*nu - 1514740*nu**2 + 889673*nu**3)*chiA)/(106920*(1 - 3*nu)**2) + (8*delta*kappaA*(-1 + nu)*nu*chiS)/(9*(1 - 3*nu)**2) - (8*kappaS*nu*(1 - 3*nu + 6*nu**2)*chiS)/(9*(1 - 3*nu)**2) + (((2208096 - 20471053*nu + 70519165*nu**2 - 101706029*nu**3 + 40204523*nu**4 + 11842250*nu**5)/(962280*(-1 + 3*nu)**3) - (8*nu*(1 - 9*nu + 12*nu**2)*chiA**2)/(9*(1 - 3*nu)**2))*chiS) - (16*delta*nu*(-9 + 46*nu + 38*nu**2)*chiA*chiS**2)/(81*(-1 + 3*nu)**3) + (8*nu*(-243 + 1269*nu - 5029*nu**2 + 5441*nu**3 + 12022*nu**4)*chiS**3)/(2187*(1 - 3*nu)**4))
)

rho[3, 3] = (
    1
    + v**2*(-7 + 4*nu)/6
    + v**3*(((-4 + 19*nu)*chiA)/(6*delta) + ((-4 + 5*nu)*chiS)/6)
    + v**5*(((-80 + 299*nu + 18*nu**2)*chiA)/(60*delta) + ((-80 + 181*nu + 94*nu**2)*chiS)/60)
    + v**4*((kappaA*(1 - 4*nu))/(2*delta) + kappaS*(sp.Rational(1, 2) - nu) + (sp.Rational(1, 2) - 2*nu)*chiA**2 + ((1 - 4*nu)*chiA*chiS)/delta + chiS**2/2)
    + v**6*((kappaS*(7 - 28*nu - 8*nu**2))/12 + (kappaA*(7 - 42*nu + 48*nu**2))/(12*delta) + ((5 - 58*nu + 95*nu**2 + 192*nu**3)*chiA**2)/(36 - 144*nu) + ((5 - 102*nu + 265*nu**2)*chiA*chiS)/(18*delta) + ((5 - 146*nu + 47*nu**2)*chiS**2)/36)
)

rho[4, 1] = (
    1
    + v*((-5*nu*chiA)/(8*delta - 16*delta*nu) + (5*nu*chiS)/(8 - 16*nu))
    + v**2*((602 - 1385*nu + 288*nu**2)/(528*(-1 + 2*nu)) + (75*nu**2*chiA**2)/(128*(1 - 2*nu)**2*(-1 + 4*nu)) + (75*nu**2*chiA*chiS)/(64*delta*(1 - 2*nu)**2) - (75*nu**2*chiS**2)/(128*(1 - 2*nu)**2))
    + v**4*(kappaS*(sp.Rational(3, 8) - 3*nu/4) + (kappaA*(3 - 18*nu - 4*nu**2))/(8*delta - 16*delta*nu) + ((25344 - 411136*nu + 2624414*nu**2 - 7125275*nu**3 + 8297344*nu**4 - 3244032*nu**5)*chiA**2)/(67584*(-1 + 2*nu)**3*(-1 + 4*nu)) + ((-25344 + 253440*nu - 598850*nu**2 + 433253*nu**3 - 54272*nu**4)*chiA*chiS)/(33792*delta*(-1 + 2*nu)**3) + ((-25344 + 95744*nu - 266718*nu**2 + 253467*nu**3 + 160640*nu**4)*chiS**2)/(67584*(-1 + 2*nu)**3))
    + v**3*((nu*(9762 - 34465*nu + 30992*nu**2)*chiA)/(4224*delta*(1 - 2*nu)**2) + (875*nu**3*chiA**3)/(1024*(-1 + 2*nu)**3*(delta - 4*delta*nu)) + (((nu*(-4482 - 479*nu + 17776*nu**2))/(4224*(1 - 2*nu)**2) + (2625*nu**3*chiA**2)/(1024*(-1 + 2*nu)**3*(-1 + 4*nu)))*chiS) + (2625*nu**3*chiA*chiS**2)/(1024*delta*(-1 + 2*nu)**3) - (875*nu**3*chiS**3)/(1024*(-1 + 2*nu)**3))
)

rho[4, 2] = (
    1
    + v**2*(1146 - 3530*nu + 285*nu**2)/(1320*(-1 + 3*nu))
    + v**3*((delta*(10 - 21*nu)*chiA)/(-15 + 45*nu) + ((10 - 59*nu + 78*nu**2)*chiS)/(-15 + 45*nu))
    + v**5*((delta*(-420 + 23816*nu - 129270*nu**2 + 184725*nu**3)*chiA)/(6600*(1 - 3*nu)**2) + ((-420 + 14984*nu - 74658*nu**2 + 96555*nu**3 + 11790*nu**4)*chiS)/(6600*(1 - 3*nu)**2))
    + v**4*((delta*kappaA)/(2 - 6*nu) + (kappaS*(1 - 2*nu + 6*nu**2))/(2 - 6*nu) + ((1 - 4*nu + 12*nu**2)*chiA**2)/(2 - 6*nu) + (delta*chiA*chiS)/(1 - 3*nu) + chiS**2/(2 - 6*nu))
)

rho[4, 3] = (
    1
    + v*((-5*nu*chiA)/(8*delta - 16*delta*nu) + (5*nu*chiS)/(8 - 16*nu))
    + v**2*((222 - 547*nu + 160*nu**2)/(176*(-1 + 2*nu)) + (75*nu**2*chiA**2)/(128*(1 - 2*nu)**2*(-1 + 4*nu)) + (75*nu**2*chiA*chiS)/(64*delta*(1 - 2*nu)**2) - (75*nu**2*chiS**2)/(128*(1 - 2*nu)**2))
    + v**4*(kappaS*(sp.Rational(3, 8) - 3*nu/4) + (kappaA*(3 - 18*nu + 12*nu**2))/(8*delta - 16*delta*nu) + ((25344 - 411136*nu + 2665694*nu**2 - 7365275*nu**3 + 8645824*nu**4 - 3244032*nu**5)*chiA**2)/(67584*(-1 + 2*nu)**3*(-1 + 4*nu)) + ((-25344 + 253440*nu - 692738*nu**2 + 808805*nu**3 - 396224*nu**4)*chiA*chiS)/(33792*delta*(-1 + 2*nu)**3) + ((-25344 + 95744*nu - 307998*nu**2 + 343707*nu**3 + 111680*nu**4)*chiS**2)/(67584*(-1 + 2*nu)**3))
    + v**3*((nu*(11298 - 43105*nu + 43088*nu**2)*chiA)/(4224*delta*(1 - 2*nu)**2) + (875*nu**3*chiA**3)/(1024*(-1 + 2*nu)**3*(delta - 4*delta*nu)) + (((nu*(-6018 + 3169*nu + 15664*nu**2))/(4224*(1 - 2*nu)**2) + (2625*nu**3*chiA**2)/(1024*(-1 + 2*nu)**3*(-1 + 4*nu)))*chiS) + (2625*nu**3*chiA*chiS**2)/(1024*delta*(-1 + 2*nu)**3) - (875*nu**3*chiS**3)/(1024*(-1 + 2*nu)**3))
)

rho[4, 4] = (
    1
    + v**2*(1614 - 5870*nu + 2625*nu**2)/(1320*(-1 + 3*nu))
    + v**3*((delta*(10 - 39*nu)*chiA)/(-15 + 45*nu) + ((10 - 41*nu + 42*nu**2)*chiS)/(-15 + 45*nu))
    + v**5*((delta*(-8280 + 42716*nu - 57990*nu**2 + 8955*nu**3)*chiA)/(6600*(1 - 3*nu)**2) + ((-8280 + 66284*nu - 176418*nu**2 + 128085*nu**3 + 88650*nu**4)*chiS)/(6600*(1 - 3*nu)**2))
    + v**4*((delta*kappaA)/2 + kappaS*(sp.Rational(1, 2) - nu) + (sp.Rational(1, 2) - 2*nu)*chiA**2 + delta*chiA*chiS + chiS**2/2)
)

rho[5, 1] = (
    1
    + v**3*(-((10 - 46*nu + 38*nu**2)*chiA)/(15*delta - 30*delta*nu) + ((10 - 54*nu + 54*nu**2)*chiS)/(-15 + 30*nu))
    + v**4*(kappaS*(sp.Rational(1, 2) - nu) + (kappaA*(5 - 30*nu - 8*nu**2))/(10*delta - 20*delta*nu) + (sp.Rational(1, 2) - 2*nu)*chiA**2 + ((5 - 30*nu - 8*nu**2)*chiA*chiS)/(5*delta - 10*delta*nu) + chiS**2/2)
)

rho[5, 2] = (
    1
    + v*((-3*delta*nu*chiA)/(5*(1 - 5*nu + 5*nu**2)) + (3*(1 - 2*nu)*nu*chiS)/(5*(1 - 5*nu + 5*nu**2)))
    + v**2*((-15828 + 84679*nu - 104930*nu**2 + 21980*nu**3)/(13650*(1 - 5*nu + 5*nu**2)) + (18*nu**2*(-1 + 4*nu)*chiA**2)/(25*(1 - 5*nu + 5*nu**2)**2) + (36*delta*(1 - 2*nu)*nu**2*chiA*chiS)/(25*(1 - 5*nu + 5*nu**2)**2) - (18*(1 - 2*nu)**2*nu**2*chiS**2)/(25*(1 - 5*nu + 5*nu**2)**2))
    + v**3*(-(delta*(9100 - 243364*nu + 1213877*nu**2 - 1907465*nu**3 + 963865*nu**4)*chiA)/(68250*(1 - 5*nu + 5*nu**2)**2) + (162*delta*nu**3*(-1 + 4*nu)*chiA**3)/(125*(1 - 5*nu + 5*nu**2)**3) + (((-9100 + 25086*nu + 17105*nu**2 + 471681*nu**3 - 2079455*nu**4 + 1859270*nu**5)/(68250*(1 - 5*nu + 5*nu**2)**2) + (486*nu**3*(1 - 6*nu + 8*nu**2)*chiA**2)/(125*(1 - 5*nu + 5*nu**2)**3))*chiS) - (486*delta*(1 - 2*nu)**2*nu**3*chiA*chiS**2)/(125*(1 - 5*nu + 5*nu**2)**3) - (162*nu**3*(-1 + 2*nu)**3*chiS**3)/(125*(1 - 5*nu + 5*nu**2)**3))
)

rho[5, 3] = (
    1
    + v**3*(-((10 - 54*nu + 62*nu**2)*chiA)/(15*delta - 30*delta*nu) + ((10 - 46*nu + 46*nu**2)*chiS)/(-15 + 30*nu))
    + v**4*(kappaS*(sp.Rational(1, 2) - nu) + (kappaA*(5 - 30*nu + 8*nu**2))/(10*delta - 20*delta*nu) + (sp.Rational(1, 2) - 2*nu)*chiA**2 + ((5 - 30*nu + 8*nu**2)*chiA*chiS)/(5*delta - 10*delta*nu) + chiS**2/2)
)

rho[5, 4] = (
    1
    + v*((-3*delta*nu*chiA)/(5*(1 - 5*nu + 5*nu**2)) + (3*(1 - 2*nu)*nu*chiS)/(5*(1 - 5*nu + 5*nu**2)))
    + v**2*((-17448 + 96019*nu - 127610*nu**2 + 33320*nu**3)/(13650*(1 - 5*nu + 5*nu**2)) + (18*nu**2*(-1 + 4*nu)*chiA**2)/(25*(1 - 5*nu + 5*nu**2)**2) + (36*delta*(1 - 2*nu)*nu**2*chiA*chiS)/(25*(1 - 5*nu + 5*nu**2)**2) - (18*(1 - 2*nu)**2*nu**2*chiS**2)/(25*(1 - 5*nu + 5*nu**2)**2))
    + v**3*(-(delta*(9100 - 268024*nu + 1386497*nu**2 - 2296805*nu**3 + 1268785*nu**4)*chiA)/(68250*(1 - 5*nu + 5*nu**2)**2) + (162*delta*nu**3*(-1 + 4*nu)*chiA**3)/(125*(1 - 5*nu + 5*nu**2)**3) + (((-9100 + 426*nu + 239045*nu**2 - 174699*nu**3 - 1436855*nu**4 + 1690430*nu**5)/(68250*(1 - 5*nu + 5*nu**2)**2) + (486*nu**3*(1 - 6*nu + 8*nu**2)*chiA**2)/(125*(1 - 5*nu + 5*nu**2)**3))*chiS) - (486*delta*(1 - 2*nu)**2*nu**3*chiA*chiS**2)/(125*(1 - 5*nu + 5*nu**2)**3) - (162*nu**3*(-1 + 2*nu)**3*chiS**3)/(125*(1 - 5*nu + 5*nu**2)**3))
)

rho[5, 5] = (
    1
    + v**3*(-((2 - 14*nu + 22*nu**2)*chiA)/(3*delta - 6*delta*nu) + ((2 - 6*nu + 6*nu**2)*chiS)/(-3 + 6*nu))
    + v**4*((kappaA*(1 - 4*nu))/(2*delta) + kappaS*(sp.Rational(1, 2) - nu) + (sp.Rational(1, 2) - 2*nu)*chiA**2 + ((1 - 4*nu)*chiA*chiS)/delta + chiS**2/2)
)

rho[6, 1] = (
    1
    + v*((7*nu*chiA)/(12*delta*(-1 + nu)) + (7*nu*chiS)/(12 - 36*nu))
    + v**2*((-245*nu**2*chiA**2)/(288*delta**2*(-1 + nu)**2) + (245*nu**2*chiA*chiS)/(144*delta*(1 - 4*nu + 3*nu**2)) - (245*nu**2*chiS**2)/(72*(2 - 6*nu)**2))
)

rho[6, 2] = (
    1
    + v**3*(-(delta*(14 - 53*nu + 36*nu**2)*chiA)/(21*(1 - 5*nu + 5*nu**2)) + ((-14 + 115*nu - 278*nu**2 + 174*nu**3)*chiS)/(21*(1 - 5*nu + 5*nu**2)))
)

rho[6, 3] = rho[6, 1]

rho[6, 4] = (
    1
    + v**3*(-(delta*(14 - 65*nu + 60*nu**2)*chiA)/(21*(1 - 5*nu + 5*nu**2)) + ((-14 + 103*nu - 230*nu**2 + 150*nu**3)*chiS)/(21*(1 - 5*nu + 5*nu**2)))
)

rho[6, 5] = rho[6, 1]

rho[6, 6] = (
    1
    + v**3*(-(delta*(14 - 85*nu + 100*nu**2)*chiA)/(21*(1 - 5*nu + 5*nu**2)) + ((-14 + 83*nu - 150*nu**2 + 110*nu**3)*chiS)/(21*(1 - 5*nu + 5*nu**2)))
)

rho[7, 2] = (
    1
    + v*((4*delta*(1 - 2*nu)*nu*chiA)/(7*(-1 + 7*nu - 14*nu**2 + 7*nu**3)) - (4*nu*(1 - 4*nu + 2*nu**2)*chiS)/(7*(-1 + 7*nu - 14*nu**2 + 7*nu**3)))
)

rho[7, 4] = rho[7, 2]
rho[7, 6] = rho[7, 2]

# --------------------------------------------------
# delta_lm phase corrections
# --------------------------------------------------

delta_phase[2, 1] = (
    -25*v**5*nu/2
    + 2*omegaE/3
    + (-((68 + 69*nu)*chiA)/(140*delta) + (-sp.Rational(17, 35) - 41*nu/28)*chiS) * omegaE**2
)

delta_phase[2, 2] = (
    7*omegaE/3
    + ((-4*delta*chiA)/3 + (4*(-1 + 2*nu)*chiS)/3) * omegaE**2
)

delta_phase[3, 1] = (
    13*omegaE/30
    + (((61 - 45*nu)*chiA)/(20*delta) + ((61 + 77*nu)*chiS)/20) * omegaE**2
)

delta_phase[3, 2] = (
    v**4 * ((4*delta*chiA)/(1 - 3*nu) + (4*(5 - 25*nu + 9*nu**2)*chiS)/(5*(1 - 3*nu)**2))
    + v**5 * ((-16*delta*nu*chiA*chiS)/(1 - 3*nu)**2 + (16*nu*(5 - 25*nu + 9*nu**2)*chiS**2)/(5*(-1 + 3*nu)**3))
    + ((10 + 33*nu)*omegaE)/(15 - 45*nu)
)

delta_phase[3, 3] = (
    13*omegaE/10
    + (((-2187 + 7339*nu)*chiA)/(540*delta) + (-sp.Rational(81, 20) + 593*nu/108)*chiS) * omegaE**2
)

delta_phase[4, 1] = (
    v**4 * (((11 - 55*nu + 1599*nu**2)*chiA)/(12*delta*(1 - 2*nu)**2) + ((11 - 33*nu - 1511*nu**2)*chiS)/(12*(1 - 2*nu)**2))
    + ((2 + 507*nu)*omegaE)/(10 - 20*nu)
)

delta_phase[4, 3] = (
    v**4 * (((891 + nu*(-7815 + 17999*nu))*chiA)/(324*delta*(1 - 2*nu)**2) + ((891 + nu*(-6033 + 2569*nu))*chiS)/(324*(1 - 2*nu)**2))
    + ((486 + 4961*nu)*omegaE)/(810 - 1620*nu)
)