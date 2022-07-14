from __future__ import annotations

from time import time

from lmfit import (Minimizer, Parameters, conf_interval, minimize, printfuncs,
                   report_errors)
from scipy import arange, array, sqrt

from bluebonnet.flow import FlowPropertiesMarder, SinglePhaseReservoirMarder
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def obfun(params, Days, Gas, pvt_gas, PressureTime):
    tau = params["tau"].value
    M = params["M"].value
    Pi = params["Pi"].value
    Pf = Pi
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
    return M * Model - Gas

def FitProductionWithPressure(D,
                                pvt,
                                NTimes, 
                                Pi, 
                                PiFit=True, 
                                FilterPressure=True,
                                Nave=30,
                                PiMax=14000,
                                MMax=100000,
                                MMin=10, 
                                NonZeroDaysOnly=True, 
                                params=None
                                  ):

    """
    D must be a data frame containing columns 'Days', 'Gas', and 'Pressure'
    pvt has information on equation of state, for example from BuildPVT 
    NTimes has number of times to iterate until stabilizes. Try 200
    Pi has initial reservoir pressure. Either fixed value or initial guess
    PiFit: Allow initial reservoir pressure to vary as fitting parameter
    FilterPressure: Run pressure through averaging filter
    Nave: Number of days to average pressure
    PiMax:Maximum allowed initial reservoir pressure. pvt had better include this pressure
    MMax: Maximum allowed M
    MMin: Minimum allowed M
    NonzeroDaysOnly: Filter out days without gas production or pressure value
    params: This is what the routine returns, and you can pass in results from previous fit. Fits tau, M, Pi
    """
    if NonZeroDaysOnly:
        Data=D[(D['Gas']>0) & (pd.notna(D['Pressure']))][['Days','Gas','Pressure']]
    else:
        Data=D[['Days','Gas','Pressure']]

    GoodDays = np.array(Data['Days'])
    Days=np.arange(0,len(GoodDays))
    PressureTime=np.array(Data['Pressure'])
    #
    if FilterPressure:
        PressureTime=sp.ndimage.filters.uniform_filter1d(PressureTime, size=Nave)
    #
    Gas=np.cumsum(np.array(Data['Gas'])) #Cumulative Gas production

    if (params==None):
        params=Parameters()
        params.add('tau',value=1000.0,min=30.0,max=Days[-1]*2) #Days
        params.add('M',value=Gas[-1],min=Gas[-2],max=MMax) #MMcf
        params.add('Pi',value=Pi,min=max(PressureTime),max=PiMax) #MMcf
    mini=Minimizer(obfun,params,fcn_args=(Days,Gas,pvt,PressureTime))
    result=mini.minimize(method='Nelder',max_nfev=NTimes)

    return(result)

def PlotProductionComparison(D,
                                pvt,
                                params,
                                FilterPressure=True,
                                Nave=30,
                                NonZeroDaysOnly=True, 
                                PlotFileName="ProductionComparison.pdf",
                                WellName="Well Name",
                                ProductionLabel="Cumulative Production",
                                PressureLabel="Pressure (psi)"):
    """
    D must be a data frame containing columns 'Days', 'Gas', and 'Pressure'
    pvt has information on equation of state, for example from BuildPVT 
    FilterPressure: Run pressure through averaging filter
    Nave: Number of days to average pressure
    NonzeroDaysOnly: Filter out days without gas production or pressure value
    params: Returned from fit
    PlotFileName: Name of plot file. If not entered, uses a default value
    WellName: Name of well to use in plot file
    ProductionLabel: Y axis label for production plot
    """    
    if NonZeroDaysOnly:
        Data=D[(D['Gas']>0) & (pd.notna(D['Pressure']))][['Days','Gas','Pressure']]
    else:
        Data=D[['Days','Gas','Pressure']]

    GoodDays = np.array(Data['Days'])
    Days=np.arange(0,len(GoodDays))
    PressureTime=np.array(Data['Pressure'])
    #
    if FilterPressure:
        PressureTime=sp.ndimage.filters.uniform_filter1d(PressureTime, size=Nave)
    #
    Gas=np.cumsum(np.array(Data['Gas'])) #Cumulative Gas production
    plt.rcParams['text.usetex'] = True
    M=params['M'].value
    tau=params['tau'].value
    Pi=params['Pi'].value
    Pf=Pi
    
    flow_propertiesM = FlowPropertiesMarder(pvt, Pi)
    res_realgasM = SinglePhaseReservoirMarder(80, Pf, Pi, flow_propertiesM)
    res_realgasM.simulate(Days/tau,pressure_fracface=PressureTime)
    print('tau={:7.5g}, Pi={:7.5g} and M={:8.5g}'.format(tau,Pi,M))
    
    rf2M = res_realgasM.recovery_factor()
    fig, (ax1,ax2) = plt.subplots(2,1)
    fig.set_size_inches(5, 6)
    ax1.plot(Days/tau, rf2M,"--", label=r"Production; $\tau=${:7.5g}, $\mathcal M=${:7.5g}".format(tau,M))
    ax1.plot(Days/tau, Gas/M, label=WellName)
    ax1.legend()
    ax1.set(xlabel="Time", ylabel=ProductionLabel, ylim=(0,None), xscale='squareroot', xlim=(0,None))
    
    ax2.plot(Days/tau, PressureTime, label=PressureLabel)
    ax2.legend()
    ax2.set(xlabel="Time", ylabel="Pressure", ylim=(0,None), xscale='squareroot', xlim=(0,None))
    
    pp=PdfPages(PlotFileName)
    pp.savefig(fig)
    pp.close()
    
