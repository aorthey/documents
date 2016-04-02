import numpy as np
import pylab as plt
import math
from matplotlib.widgets import Slider

class InterpolateQuintic():
        coeff = []
        duration = 0.0
        knots = 0
        tangent_length = 0.07
        xmean = 0.0
        ymean = 0.0
        def __init__(self, p, dp, ddp):
                assert(p.ndim>1)
                assert(p.shape[1]>1)
                Ndim = p.shape[0]
                Nwaypoints = p.shape[1]
                self.xmean = np.mean(p[0,:])
                self.ymean = np.mean(p[1,:])
                self.coeff = np.zeros((Nwaypoints-1,6,Ndim))
                self.knots = Nwaypoints-1

        def Interpolate(self,T):
                self.duration = T
                self.tend = T*self.knots
                for i in range(0,self.knots):
                        [A,B,C,D,E,F] = self.interpolate_wpt(T,\
                                        p[:,i],dp[:,i],ddp[:,i], \
                                        p[:,i+1],dp[:,i+1],ddp[:,i+1])
                        self.coeff[i,:,:]=np.array((A,B,C,D,E,F))

        def getCurCoeff(self,dt):
                assert(dt>=0)
                assert(dt<=self.tend)
                if dt>=(self.tend):
                        curknot = self.knots-1
                else:
                        curknot = int(math.floor(dt/self.duration))
                [A,B,C,D,E,F] = self.coeff[curknot,:,:]
                t = dt-curknot*self.duration
                return [t,A,B,C,D,E,F]

        def Eval(self,dt):
                [t,A,B,C,D,E,F] = self.getCurCoeff(dt)
                return A + B*t + C*t**2 + D*t**3 + E*t**4 + F*t**5
        def Evald(self,dt):
                [t,A,B,C,D,E,F] = self.getCurCoeff(dt)
                return B + 2*C*t + 3*D*t**2 + 4*E*t**3 + 5*F*t**4
        def Evaldd(self,dt):
                [t,A,B,C,D,E,F] = self.getCurCoeff(dt)
                return 2*C + 6*D*t + 12*E*t**2 + 20*F*t**3

        def Info(self):
                print "-----------------------------------"
                print "Trajectory -- Quintic Interpolation"
                print "-----------------------------------"
                print "Duration    : ",self.tend
                print "Startpoint  : ",self.Eval(0)
                print "Goalpoint   : ",self.Eval(self.tend)
                print "-----------------------------------"

        def interpolate_wpt(self,T,p0,dp0,ddp0,p1,dp1,ddp1):
                Ndim = p0.shape[0]
                A = np.zeros(Ndim)
                B = np.zeros(Ndim)
                C = np.zeros(Ndim)
                D = np.zeros(Ndim)
                E = np.zeros(Ndim)
                F = np.zeros(Ndim)

                TT = T*T
                TTT = T*T*T

                A=p0
                B=dp0
                C=0.5*ddp0

                U1 = (p1-p0-dp0*T-0.5*ddp0*TT)/TTT
                U2 = (dp1-dp0-ddp0*T)/TT
                U3 = (ddp1-ddp0)/T

                D = (10*U1 - 4*U2 + 0.5*U3)
                E = (-15*U1 + 7*U2 - U3)/T
                F = (6*U1 - 3*U2 + 0.5*U3)/TT

                return [A,B,C,D,E,F]

        def Plot(self):
                fig = plt.figure(facecolor='white')
                np.set_printoptions(precision=4)
                T0 = 0.1

                M=100
                tvec = np.linspace(0,self.knots*T0,M)
                Y = np.array(map(lambda t: self.Eval(t), tvec))
                l, = plt.plot(Y[:,0],Y[:,1],'-r',linewidth=3)

                tvec = np.linspace(0,self.knots*self.duration,self.knots+1)
                K = np.array(map(lambda t: self.Eval(t), tvec))
                plt.plot(K[:,0],K[:,1],'ok',markersize=10)
                tvec = np.linspace(0,self.knots*self.duration,self.knots+1)
                dK = np.array(map(lambda t: self.Evald(t), tvec))

                for i in range(0,self.knots+1):
                        v = dK[i,:]/np.linalg.norm(dK[i,:])
                        V1 = K[i,:] + v*self.tangent_length
                        V2 = K[i,:] - v*self.tangent_length
                        V = np.vstack((V1,V2)).T
                        plt.plot(V[0,:],V[1,:],'-k',linewidth=3)

                plt.title("Quintic Interpolation $a_0+a_1t+a_2t^2+a_3t^3+a_4t^4+a_5t^5$", \
                                fontsize=20,y=1.02)
                plt.axis('equal')
                plt.xlabel('X',fontsize=22)
                plt.ylabel('Y',fontsize=22)

                axcolor = 'lightgoldenrodyellow'
                axfreq = plt.axes([0.25, 0.12, 0.5, 0.03], axisbg=axcolor)
                tduration = Slider(axfreq, 'Smoothing', 0.01, 0.5, color='black',valinit=T0)

                def update(val):
                        T = tduration.val
                        M=100
                        self.Interpolate(T)
                        tvec = np.linspace(0,self.knots*T,M)
                        Y = np.array(map(lambda t: self.Eval(t), tvec)).T
                        l.set_xdata(Y[0,:])
                        l.set_ydata(Y[1,:])
                        fig.canvas.draw_idle()

                tduration.on_changed(update)
                plt.show()


if __name__ == '__main__':
        
        duration = 0.15
        S = 2.0
        p = S*np.array(((0,0),(0.1,0.2),(0.2,0.4),(0.3,0.3),(0.4,0.5),(0.5,0.1))).T
        dp = S*np.array(((1,0.2),(0.1,1),(1,0),(0.5,0.2),(0.5,-0.2),(1,-0.5))).T
        ddp = S*np.array(((1,0.5),(0.5,1),(1,0.5),(1.5,0.5),(1.2,-1.2),(1,0))).T
        path = InterpolateQuintic(p,dp,ddp)
        path.Interpolate(duration)
        path.Info()
        path.Plot()
