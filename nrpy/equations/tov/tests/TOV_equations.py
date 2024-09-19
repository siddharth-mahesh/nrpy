from mpmath import mpc, mpf  # type: ignore

trusted_dict = {
    "dM_dr": mpf("0.0181392176313686643390698883937952"),
    "dP_dr": mpf("4.66855058947693486519180142696659"),
    "dnu_dr": mpf("-13.9032468056448406461804958524378"),
    "dr_iso_dr": mpc(real="0.0", imag="-0.373370739320064218403895409189262"),
}
