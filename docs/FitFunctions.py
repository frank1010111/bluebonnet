from __future__ import annotations

from time import time

import matplotlib.pyplot as plt
from lmfit import (Minimizer, Parameters, conf_interval, minimize, printfuncs,
                   report_errors)
from scipy import arange, array, sqrt

from bluebonnet.flow import FlowPropertiesMarder, SinglePhaseReservoirMarder


def obfun(params, Days, Gas, pvt_gas, PressureTime):
    tau = params["tau"].value
    M = params["M"].value
    Pi = params["Pi"].value
    Pf = Pi
    # t0=params['t0'].value
    # t0=0
    t = Days / tau
    t0 = time()
    flow_propertiesM = FlowPropertiesMarder(pvt_gas, Pi)
    res_realgasM = SinglePhaseReservoirMarder(80, Pf, Pi, flow_propertiesM)
    res_realgasM.simulate(t, pressure_fracface=PressureTime)
    rf2M = res_realgasM.recovery_factor()
    print(
        "Simulation took {:5.2g} s; tau is {:7.5g}, Pi is {:7.5g} and M is {:7.5g}".format(
            time() - t0, tau, Pi, M
        )
    )
    Model = rf2M
    # print ('M is',M, 'Model is', Model,'and C is',C)
    # print ('M is',M,' tau is ',tau)
    # print('rms    ',sqrt((((M*Model-C))**2).mean()))
    return M * Model - Gas


def CheckRefrack(Rates, Months):
    Sqrt = 1.0 / sqrt(arange(1, Months + 1, 1))
    Refrack = False
    params = Parameters()
    params.add("M", value=Rates[12], min=0)
    for i in range(12, len(Rates) - Months):
        R = Rates[i : i + Months]
        result = minimize(SqrtObfun, params, args=(R, Sqrt), method="leastsq")
        M = result.params["M"].value
        dM = result.params["M"].stderr
        if (dM / M) < 0.05:
            print("Found refrack at month ", i)
            print(printfuncs.fit_report(result.params))
            Refrack = True
    return Refrack


def FitData(
    DataIndex,
    Times,
    Cums,
    ScaledCumulative,
    plot="Y",
    FirstMonth=3,
    tau0=10,
    M0=0.01,
    MMax=20000,
    tlimit=240,
    CI=False,
    t0=0.5,
    API="",
    method="Nelder",
    PlotFileName="test1.pdf",
    ylabel="Production (MMcf)",
):
    if API == "":
        k = list(Times.keys())
        # print(Times[k[DataIndex]])
        T = array(Times[k[DataIndex]])[FirstMonth:-1] - t0
        C = array(Cums[k[DataIndex]])[FirstMonth:-1]
        API = k[DataIndex]
    else:
        k = list(Times.keys())
        T = array(Times[API])[FirstMonth:-1] - t0
        C = array(Cums[API])[FirstMonth:-1]
        DataIndex = k.index(API)
    params = Parameters()
    # print('T= ',T)
    params.add("tau", value=tau0, min=T[-1] / 10.0, max=tlimit)  # Months
    params.add("M", value=C[-1], min=M0, max=MMax)  # MMcf
    # params.add('t0',value=1,min=0,max=2) #Months
    # t0=0
    dt0 = 0
    mini = Minimizer(
        obfun,
        params,
        fcn_args=(Times, Cums, DataIndex, ScaledCumulative, t0, FirstMonth),
    )
    # result=minimize(obfun,params,args=(Times,Cums,DataIndex,ScaledCumulative,t0),method='Nelder')
    # result=mini.minimize(method='Nelder')
    result = mini.minimize(method=method)
    result = mini.minimize(method="leastsq", params=result.params)
    M = result.params["M"].value
    dM = result.params["M"].stderr
    # t0=params['t0'].value
    # dt0=params['t0'].stderr
    tau = result.params["tau"].value
    dtau = result.params["tau"].stderr
    chisqr = result.chisqr / C[-1] ** 2
    rms = sqrt(chisqr / (len(C) - 1))
    t_max = (T[-1]) / tau
    try:
        correl = result.params["tau"].correl["M"]
    except:
        correl = ""
    if CI == True:
        # result.leastsq()
        # print(dM)
        # print(printfuncs.fit_report(result.params))
        # ci=conf_interval(mini,result)
        # printfuncs.report_ci(ci)
        # print('ci is', ci['M'])
        try:
            ci = conf_interval(mini, result)
            # print(printfuncs.report_ci(ci))
            dtau = ci["tau"][3][1] - ci["tau"][2][1]
            dM = ci["M"][3][1] - ci["M"][2][1]
        except:
            print("CI failed for ", API, " ", dtau, dM)
            dtau = dtau
            dM = dM
            dtau = 0
            dM = 0  # If the fancy routine can't find anything, list uncertainty as zero
        # dt0=ci['t0'][3][1]-ci['t0'][2][1]
    if plot == "Y":
        report_errors(result.params)
        print("t_max is", t_max)
        print("chisquare is", chisqr, "and rms is ", rms)
        toff = T[0]
        coff = C[0]
        toff = 0
        coff = 0
        CFit = C - coff + result.residual
        Model = M * ScaledCumulative((T - toff) / tau)
        plt.clf()
        plt.plot(sqrt(T - toff), C - coff, "k")
        plt.plot(sqrt(T - toff), CFit, "r")

        plt.rc("text", usetex=True)
        plt.xlabel(r"Square root time $\sqrt{t}$,  time in months")
        plt.ylabel(ylabel)
        # plt.plot(sqrt(T-T[0]),Model,'g')
        plt.savefig(PlotFileName)
        # plt.clf()
        # plt.plot(sqrt(T/T[-1]),C/CFit[-1],'k')
        # plt.plot(sqrt(T/T[-1]),sqrt(T/T[-1]),'r')
        # plt.plot(sqrt(T-T[0]),Model,'g')
    # plt.savefig('test2.pdf')
    return API, tau, dtau, M, dM, t0, dt0, t_max, chisqr, rms, correl, result
