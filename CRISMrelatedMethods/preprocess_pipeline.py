
from CRISMrelatedMethods.SFC import *
from CRISMrelatedMethods.preprocessing import * 

def getPreProcessed(sourceIF,prepSteps='',cr_deg=5,smParams={},CRParams={},CCParams={},doPrint=False):
    if sourceIF[0]==np.Inf: return sourceIF
    SRstdRatio=.25 if smParams.get('stdRatio') is None else smParams['stdRatio']
    SRwindowSize=5 if smParams.get('SRwindowSize') is None else smParams['SRwindowSize']
    smWindow=5 if smParams.get('smWindow') is None else smParams['smWindow']
    smOrder=2 if smParams.get('smOrder') is None else smParams['smOrder']
    cr_deep=25 if CRParams.get('deep') is None else CRParams['deep']
    cr_method='div' if CRParams.get('method') is None else CRParams['method']
    cr_curveFitMethod='poly' if CRParams.get('curveFitMethod') is None else CRParams['curveFitMethod']
    cr_continuumType='curve' if CRParams.get('continuumType') is None else CRParams['continuumType']
    cr_wavelengths=np.arange(sourceIF.shape[0]) if CRParams.get('wavelengths') is None else CRParams['wavelengths']
    CCHorder=1 if CCParams.get('order') is None else CCParams['order']
    CCHcurve=False if CCParams.get('curve') is None else CCParams['curve']
    CCHsegMaximaLimit=0 if CCParams.get('segMaximaLimit') is None else CCParams['segMaximaLimit']
    cch_CR=False if CCParams.get('CR') is None else CCParams['CR']
    
    if doPrint: print('prepSteps,SRstdRatio,smWindow,smOrder,cr_wavelengths,cch_CR,CCHorder,CCHcurve')
    if doPrint: print(prepSteps,SRstdRatio,smWindow,smOrder,cr_wavelengths,cch_CR,CCHorder,CCHcurve)
#     print('sourceIF',sourceIF)

    thisSpectra=sourceIF
    for m in range(0,len(prepSteps),2):
        thisStep=prepSteps[m:m+2]
        if thisStep=='cr':
            thisSpectra=getMYCRspectra(thisSpectra,deg=cr_deg,deep=cr_deep,method=cr_method,wavelengths=cr_wavelengths,curveFitMethod=cr_curveFitMethod,continuumType=cr_continuumType)
        if thisStep=='SH':
            thisSpectra=getCRsuh(thisSpectra,cr_wavelengths,deep=cr_deep)
        if thisStep=='CC':
#             print('before CC',thisSpectra)
            thisSpectra=getConvexConaveUH2(thisSpectra,targetWL=cr_wavelengths,maximalimit=CCHsegMaximaLimit,CR=cch_CR,doPrint=doPrint)
#             print('after CC',thisSpectra)
        if thisStep=='CR':
#             try:
#                 thisSpectra=getContinumRemovedSpectra(thisSpectra)
                thisSpectra=thisSpectra/getUpperCH(thisSpectra,targetWL=cr_wavelengths)
#             except:
#                 thisSpectra=np.ones((1,len(spectralWavelengthSet)))
        elif thisStep=='SS':
#             print(thisSpectra)
            thisSpectra=getStandardScaledSpectra(thisSpectra)
        elif thisStep=='MM':
            thisSpectra=getMinMaxScaledSpectra(thisSpectra)
        elif thisStep=='sm':
            thisSpectra=savitzky_golay(thisSpectra,window_size=smWindow, order=smOrder)
        elif thisStep=='SR':
            thisSpectra=removeSpikes(thisSpectra,stdRatio=SRstdRatio,windowSize=SRwindowSize)
#             thisSpectra=getMovingMedianFilter(thisSpectra,stdRatio=SRstdRatio)
    return thisSpectra