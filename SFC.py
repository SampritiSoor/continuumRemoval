import numpy as np
from preprocessing import * 
#===============================================================================================
def getFitcontiPoints2(seg):
#     print(seg)
    x_s=seg['start'][1]
    x_t=seg['end'][1]
    
    x=np.array([seg['maximaList'][i][1] for i in range(len(seg['maximaList']))])
    y=np.array([seg['maximaList'][i][2] for i in range(len(seg['maximaList']))])
    
    a=(y-1)/((x-x_s)*(x-x_t))
    curve=((np.matmul(a.reshape(x.shape[0],1),((x-x_s)*(x-x_t)).reshape(1,x.shape[0]))+1)).astype(np.float32)
    y_=(np.repeat(np.array([y]),y.shape[0],axis=0)).astype(np.float32)
    c=np.sum(np.where(np.round((curve-y_),7)<0,True,False),axis=1)
    for b in np.argsort(-a):
        if c[b]==0: break
            
    plt.scatter(x,y,color='b')
    plt.scatter(x,a[b]*(x-x_s)*(x-x_t)+1,color='r')
    plt.plot(x,y,color='b')
    plt.plot(x,a[b]*(x-x_s)*(x-x_t)+1,color='r')
    plt.show()
    return a[b]*(seg['wavelengths']-x_s)*(seg['wavelengths']-x_t)+1

def getFitcontiPoints(seg):
#     print(seg)
    x_s=seg['start'][1]
    x_t=seg['end'][1]
    x=np.array([seg['maximaList'][i][1] for i in range(len(seg['maximaList']))])
    y=np.array([seg['maximaList'][i][2] for i in range(len(seg['maximaList']))])
#     x=np.array(seg['wavelengths'])
#     y=np.array(seg['IFs'])
    c=1

    aa=np.sum(y*(x-x_t)*(x-x_s))-c*np.sum((x-x_t)*(x-x_s))
    bb=np.sum((x-x_t)*(x-x_s)*(x-x_t)*(x-x_s))
    a=aa/bb
    
    y_=a*(seg['wavelengths']-x_t)*(seg['wavelengths']-x_s)+c
#     y_=np.where(y_<seg['IFs'],seg['IFs'],y_)  #**********************        

    return y_

def getConvexConaveUH2(aSpectra,targetWL=None,doPrint=False,doPlot=False,returnPlot=False,order=1,Points=False,maximalimit=0,CR=False,hull=False):
    if returnPlot:
        P=[]
#     print('maximalimit',maximalimit)
    if targetWL is None: targetWL=np.arange(aSpectra.shape[0])
    aSpectra=aSpectra.astype(np.float32)
    uch=getUpperCH(aSpectra,targetWL,doPrint=doPrint)
    uchDict={targetWL[w]:uch[w] for w in range(targetWL.shape[0])}
#     print('uchDict',uchDict)
    CRspectra=aSpectra/uch
    CChull=np.ones(aSpectra.shape)
    targetWLDict={targetWL[w]:w for w in range(targetWL.shape[0])}
#     print('targetWLDict',targetWLDict)
    
    segments,myHullPointsW,myHullPointsIF=getSegments(CRspectra,aSpectra,targetWL=targetWL,doPrint=doPrint)
    myHullPointsW,myHullPointsIF=[],[]
    firstSeg=True
    if doPrint: print(len(segments))
    fitP=[]
    for S in segments:
        if firstSeg:
            myHullPointsW.append(S['start'][1])
            myHullPointsIF.append(1)
            firstSeg=False

        if len(S['maximaList'])>maximalimit:
            fitPoints=getFitcontiPoints(S) #@@@@@@@@@@@
            if doPrint: print(S['start'],S['end'],[targetWLDict[S['wavelengths'][i]] for i in range(fitPoints.shape[0])],[CRspectra[S['wavelengths'][i]] for i in range(fitPoints.shape[0])])
            fitP.append(fitPoints)
            if returnPlot:
                P.append({'spec_X':[S['start'][1]]+[S['wavelengths'][i] for i in range(fitPoints.shape[0])]+[S['end'][1]],
                          'spec_Y':[1]+[CRspectra[targetWLDict[S['wavelengths'][i]]] for i in range(fitPoints.shape[0])]+[1],
                          'fit_X':[targetWL[targetWLDict[S['wavelengths'][0]]-1]]+[S['wavelengths'][i] for i in range(fitPoints.shape[0])]+[targetWL[targetWLDict[S['wavelengths'][-1]]+1]],
                          'fit_Y':[1]+list(fitPoints)+[1]
                })
            if doPlot:
                plt.plot([S['start'][1]]+[S['wavelengths'][i] for i in range(fitPoints.shape[0])]+[S['end'][1]],[1]+[CRspectra[targetWLDict[S['wavelengths'][i]]] for i in range(fitPoints.shape[0])]+[1],label='CRspectra')
                plt.plot([targetWL[targetWLDict[S['wavelengths'][0]]-1]]+[S['wavelengths'][i] for i in range(fitPoints.shape[0])]+[targetWL[targetWLDict[S['wavelengths'][-1]]+1]],[1]+list(fitPoints)+[1],label='fit')
                plt.show()
            for i in range(len(S['wavelengths'])):
                CRspectra[targetWLDict[S['wavelengths'][i]]]=CRspectra[targetWLDict[S['wavelengths'][i]]]/fitPoints[i]
                CChull[targetWLDict[S['wavelengths'][i]]]=fitPoints[i]
            if CR: #**********************        
                uch_thisSeg=getUpperCH(CRspectra[S['start'][0]:S['end'][0]+1],targetWL[S['start'][0]:S['end'][0]+1])     
                CRspectra[S['start'][0]:S['end'][0]+1]=CRspectra[S['start'][0]:S['end'][0]+1]/uch_thisSeg
                CChull[S['start'][0]:S['end'][0]+1]=uch_thisSeg
    if returnPlot:
        return P
    elif hull:
        return CChull
    else:
        return np.array(CRspectra)
#===============================================================================================

from math import atan2, pi, degrees, isclose

# def angle(C, B, A):
#     Ax, Ay = A[0]-B[0], A[1]-B[1]
#     Cx, Cy = C[0]-B[0], C[1]-B[1]
#     a = atan2(Ay, Ax)
#     c = atan2(Cy, Cx)
#     if a < 0: a += pi*2
#     if c < 0: c += pi*2
#     return np.round(degrees((pi*2 + c - a) if a > c else (c - a)),2)

def getConcaveHullPoints(seg,doPrint=False):
    hullPoints=[]
    stack=[seg['start'],seg['maximaList'][0]]
    hullPoints.append(0)
    for s in range(1,len(seg['maximaList'])):
        if doPrint: print("seg['maximaList'][s]",seg['maximaList'][s],'stack',stack)
        while True:
            A=(stack[-2][1],stack[-2][2])
            B=(stack[-1][1],stack[-1][2])
            C=(seg['maximaList'][s][1],seg['maximaList'][s][2])
            if doPrint: print(A,B,C,'angle',angle(A,B,C))
            if angle(A,B,C)<=180:
                if doPrint: print('here')
                stack.append(seg['maximaList'][s])
                hullPoints.append(s)
                break
            else:
                if len(stack)==2:
                    break
                else:
                    if stack[-2][2]>=stack[-1][2]:
                        break
                    else:
                        if stack[-2][2]>seg['maximaList'][s][2]:
                            break
                        else:
                            stack.pop(-2)
                            hullPoints.pop(-2)
            break
    return hullPoints

def getSegments(CRspectra,aSpectra,targetWL=None,doPrint=False):
    if targetWL is None: targetWL=np.arange(aSpectra.shape[0])
    myHullPointsIF=[]
    myHullPointsW=[]
    segmentStarted=False
    i=0
    segments=[]
    if doPrint: print(CRspectra.shape[0],CRspectra)
    while i<(CRspectra.shape[0]):
        if doPrint: print(i,'segmentStarted and np.isclose(np.round(CRspectra[i],4),1) and CRspectra[i-1]<1',segmentStarted , np.isclose(np.round(CRspectra[i],4),1) , CRspectra[i-1]<1)
        if np.isclose(np.round(CRspectra[i],4),1):
            myHullPointsIF.append(aSpectra[i])
            myHullPointsW.append(targetWL[i])
        if not segmentStarted and i<(CRspectra.shape[0])-1 and np.isclose(np.round(CRspectra[i],4),1) and CRspectra[i+1]<1:
            segmentStarted=True
            thisSegment={'start':(i,targetWL[i],CRspectra[i])}
            thisSegment['maximaList']=[]
            if doPrint: print(i,'start')
        elif segmentStarted and np.isclose(np.round(CRspectra[i],4),1) and CRspectra[i-1]<1:
            segmentStarted=False
            thisSegment['end']=(i,targetWL[i],CRspectra[i])
            segments.append(thisSegment)
            if doPrint: print(i,'end')
            i-=1
        elif segmentStarted:
            if CRspectra[i]>CRspectra[i-1] and CRspectra[i]>CRspectra[i+1]:
                thisSegment['maximaList'].append((i,targetWL[i],CRspectra[i],1-CRspectra[i]))
        i+=1
    for S in segments:
        S['wavelengths']=targetWL[S['start'][0]+1:S['end'][0]]
        S['IFs']=CRspectra[S['start'][0]+1:S['end'][0]]
    return segments,myHullPointsW,myHullPointsIF
    
#===============================================================================================
# import numpy as np
# from scipy.spatial import ConvexHull
# from scipy.interpolate import interp1d
from math import factorial
from sklearn.preprocessing import StandardScaler
from math import atan2, pi, degrees, isclose

def angle(C, B=(0,0), A=(1,0), nintyDegree=False):
    Ax, Ay = A[0]-B[0], A[1]-B[1]
    Cx, Cy = C[0]-B[0], C[1]-B[1]
    a = atan2(Ay, Ax)
    c = atan2(Cy, Cx)
    if a < 0: a += pi*2
    if c < 0: c += pi*2
#     deg= np.round(degrees((pi*2 + c - a) if a > c else (c - a)),2)
    deg= degrees((pi*2 + c - a) if a > c else (c - a))
#     print(deg)
    if not nintyDegree:
        return deg
    else:
        return deg if deg<=90 else deg-360

def getUpperHullPoints_CH(spectra,targetWL=None,doPrint=False,doPlot=False):
    if targetWL is None: targetWL=np.arange(spectra.shape[0])
    points=list(zip(targetWL,spectra)) 
    if doPrint: print('getUpperHullPoints_CH','points',points)
    
    hullPoints=[]
    stack=[points[0],points[1]]
    for i in range(2,len(points)):
        while True:
            if angle(stack[-2],stack[-1],points[i])>=180:
#                 if doPrint: print('here1',stack[-2:],points[i],angle(stack[-2],stack[-1],points[i]))
                stack.append(points[i])
                break
            else:
#                 if doPrint: print('here2')
                stack.pop(-1)
                if len(stack)==1:
                    stack.append(points[i])
                    break
        if doPlot:
            print(i)
            plt.figure(figsize=(10,2))
            plt.plot([s[0] for s in stack],[s[1] for s in stack])
            plt.scatter([s[0] for s in stack],[s[1] for s in stack],color='r')
            plt.plot(targetWL[:i+1],spectra[:i+1])
            plt.show()
    return [s[0] for s in stack],[s[1] for s in stack]

# def getUpperHullPoints_CH(sourceSpectra,targetWL=None):
#     if targetWL is None: targetWL=np.arange(sourceSpectra.shape[0])
#     mat = np.column_stack((targetWL,list(sourceSpectra)))
#     hull = ConvexHull(mat)
#     hullPoints=[]
#     for s in range(len(hull.simplices)):
#         for i in range(mat[hull.simplices[s]].shape[0]):
#             if list(mat[hull.simplices[s]][i,:]) not in hullPoints:
#                 hullPoints.append(list(mat[hull.simplices[s]][i,:]))
#     hullPoints.sort(key=lambda x:x[0])
#     upperHull=[]
#     currPt=hullPoints[0]
#     for h in range(len(hullPoints)):
#         if hullPoints[h][1]>=currPt[1]:
#             upperHull.append(hullPoints[h])
#             currPt=hullPoints[h]
#     for h in range(len(hullPoints)):
#         if hullPoints[h][0]<=upperHull[-1][0]:
#             continue
#         if h==len(hullPoints)-1:
#             upperHull.append(hullPoints[h])
#             continue
#         tempinterP=interp1d([hullPoints[h-1][0],hullPoints[h+1][0]],[hullPoints[h-1][1],hullPoints[h+1][1]], kind='linear')
#         interpVal=tempinterP([hullPoints[h][0]])[0]
#         if interpVal<=hullPoints[h][1]:
#             upperHull.append(hullPoints[h])
#     UHx,UHy=[],[]
#     for u in upperHull:
#         UHx.append(u[0])
#         UHy.append(u[1])
#     return UHx,UHy



def getUpperCH(sourceSpectra,targetWL=None,doPrint=False):
    if targetWL is None: targetWL=np.arange(spectra.shape[0])
    UHx,UHy=getUpperHullPoints_CH(sourceSpectra,targetWL=targetWL,doPrint=doPrint) #
    if doPrint: print("getUpperCH",'UHx,UHy',UHx,UHy,type(UHy[0]))
#     UHinterpFunc=interp1d(UHx,UHy, kind='linear')
#     interpUH=(UHinterpFunc(targetWL)).astype(np.float32)
    interpUH=(myInterpolation(UHx,UHy,targetWL,doPrint=doPrint)).astype(np.float32)
    if doPrint: print("getUpperCH",'interpUH',list(interpUH))
    return interpUH
def getContinumRemovedSpectra_CH(sourceSpectra,targetWL=None):
    continumRemoved=sourceSpectra/getUpperCH(sourceSpectra,targetWL)
    return continumRemoved

def getMinMaxScaledSpectra(spectra):
    return (spectra-np.min(spectra))/(np.max(spectra)-np.min(spectra))

def getStandardScaledSpectra(spectra):
    targetSpectra=spectra.reshape(-1,1)
    scaler=StandardScaler()
    targetSpectra=scaler.fit_transform(targetSpectra)
    targetSpectra=(targetSpectra.reshape(1,-1))[0]
    return targetSpectra

def getSavitzkyGolaySmoothing(y, window_size=5, order=2, deriv=0, rate=1):
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def myInterpolation(sourceX,sourceY,targetX,doPrint=False):
    # print code is wrong with s_i+1
    sourceX=np.array(sourceX).astype(np.float32)
    sourceY=np.array(sourceY).astype(np.float32)
    targetX=np.array(targetX).astype(np.float32)
    if targetX[0]<sourceX[0]: print('targetX[0]<sourceX[0]')
    if targetX[-1]>sourceX[-1]: print('targetX[-1]>sourceX[-1]')
        
#     for i in range(sourceY.shape[0]):
#         if sourceX[i]>targetX[0]: break
#     s_i,t_i=i-1,0
    s_i,t_i=0,0
    
    targetY=[]
    while True:
        if doPrint: print('\nmyInterpolation','s_i',s_i,'t_i',t_i,'sourceX[s_i]',sourceX[s_i],'targetX[t_i]',targetX[t_i],not (targetX[t_i]>=sourceX[s_i] and targetX[t_i]<=sourceX[s_i+1]))
        if not (targetX[t_i]>=sourceX[s_i] and targetX[t_i]<=sourceX[s_i+1]):
                s_i+=1
                continue
                
        if doPrint: print('myInterpolation','Before','s_i',s_i,'t_i',t_i,'sourceX[s_i]',sourceX[s_i],'targetX[t_i]',targetX[t_i])
        if sourceX[s_i]==targetX[t_i]: 
            targetY.append(np.float32(sourceY[s_i]))
        elif sourceX[s_i+1]==targetX[t_i]: 
            targetY.append(np.float32(sourceY[s_i+1]))
        else:
            targetY.append(np.float32(sourceY[s_i]+(sourceY[s_i+1]-sourceY[s_i])*((targetX[t_i]-sourceX[s_i])/(sourceX[s_i+1]-sourceX[s_i]))) )
        if doPrint: print('myInterpolation','s_i',s_i,'t_i',t_i,'sourceX[s_i]',sourceX[s_i],'targetX[t_i]',targetX[t_i],'sourceX[s_i+1]',sourceX[s_i+1])
        if doPrint: print('myInterpolation','sourceY[s_i+1]',sourceY[s_i+1],'sourceY[s_i]',sourceY[s_i],'targetY[-1]',targetY[-1] ,(sourceY[s_i+1]-sourceY[s_i]),(targetX[t_i]-sourceX[s_i]),(sourceX[s_i+1]-sourceX[s_i]) )
        t_i+=1
        if t_i==targetX.shape[0]: break
#         if targetX[t_i]>=sourceX[s_i+1]: s_i+=1
        if doPrint: print('myInterpolation','After','s_i',s_i,'t_i',t_i,'sourceX[s_i]',sourceX[s_i],'targetX[t_i]',targetX[t_i])
    if doPrint: print('myInterpolation','targetY',targetY)
    return np.array(targetY)

#===============================================================================================
def getGTindex(c):
    cc=c
    if c in [13,20]: cc=13
    elif c in [8,22]: cc=22
    elif c in [9,23]: cc=23
    elif c in [1,10,24]: cc=24
    return cc
def sam(a,b):
    return np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
def euclDist(v1,v2):
    return np.linalg.norm(np.array(v1) - np.array(v2)) 
def absDist(v1,v2):
    return sum(np.absolute(np.array(v1) - np.array(v2)))

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
