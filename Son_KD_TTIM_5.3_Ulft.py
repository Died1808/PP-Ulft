import math
import numpy as np
import pandas as pd
from ttim import *
from pylab import *
import matplotlib
from matplotlib import cm
import fiona
import os
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

pd.set_option('display.max_columns', None) 
pd.options.display.width=None
pd.set_option('display.max_rows', 10) 
pd.set_option('display.expand_frame_repr', True)
pd.options.display.float_format = '{:,.2f}'.format

def kleuren(n):
    cm = plt.get_cmap('gist_rainbow')
    color=[cm(1.*i/n) for i in range(n)]
    params = {'font.family': 'sans-serif',
              'font.sans-serif': 'arial',
              'axes.labelsize': 10,
              'axes.facecolor': '#ffffff', 
              'axes.labelcolor': 'black',
              'axes.prop_cycle': cycler('color', color),
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'lines.linewidth': 1,
              'grid.color': 'grey',
              'grid.linestyle': 'dashed',
              'grid.linewidth': 0.5,
              'text.usetex': False,
              'font.style': 'normal',
              'font.variant':'normal',
              'figure.facecolor': 'white',
              'font.size':8,
              'figure.autolayout': True,
              'figure.figsize': (8,6),
              'figure.dpi': 100,
              }
    plt.rcParams.update(params)

class GEF:
    def __init__(self):
        self._data_seperator = ' '
        self._columns = {}
        self.x = 0.
        self.y = 0.
        self.z = 0.
        self.dz = []
        self.qc = []
        self.pw = []
        self.wg = []
        self.c  = []
        self.kb = []
        self.dist =[]
        self.npor=[]
        
    def readFile(self, filename):
        lines = open(filename, 'r').readlines()
        for line in lines:
            reading_header = True
        for line in lines:   
            if reading_header:
                self._parseHeaderLine(line)
            else:
                self._parseDataLine(line)
            if line.find('#EOH') > -1:
                if self._check_header():
                    reading_header = False
                else:
                    print(filename,'bestaat al')
                    return
            
    def _check_header(self):
        if not 1 in self._columns:
            return False
        if not 2 in self._columns:
            return False
        return True

    def _parseHeaderLine(self, line):
        for xe in ['#COMMENT', 'Peil=', 'uitvoerder', 'materieel','WATERSTAND',
                    'opmerkingen#MEASUREMENTTEXT','==','= NAP','antropogeen']:
            if xe in line:
                return          
        if len(line.split()) == 0:
            return
        
        keyword, argline = line.split('=')         
        keyword = keyword.strip()
        argline = argline.strip()
        args = argline.split(',')
      
        if '#XYID' in line:
            argline = argline.replace('.','')        
        args = argline.split(',')

        if 'Waterspanning'  in line:
            self.u = float(args[3])
        if 'Waterdruk'  in line:
            self.u = float(args[3]) 
        try:
            if 'waterspanning'  in line:
                self.u = int(args[3])
        except ValueError:
            return

        if keyword=='#XYID':
            if float(args[1]) < 1e5:
                args[1] = args[1]
            else:
                args[1]=args[1].replace('.','')
            self.x = float(args[1])
            args[2]=args[2].replace('.','')
            self.y = float(args[2])
            if (len(str(int(self.x))))>5:
                 self.x=int(self.x/pow(10,len(str(int(self.x)))-6))
            if (len(str(int(self.y))))>5:
                 self.y=int(self.y/pow(10,len(str(int(self.y)))-6))
            if self.x > 3e5:
                self.x=self.x/10

        elif keyword=='#ZID':
            self.z = round(float(args[1]),3)
           
        elif keyword=='#COLUMNINFO':
            column = int(args[0])
            dtype = int(args[-1])
            if dtype==11:
                dtype = 10
            self._columns[dtype] = column - 1    
       
    def _parseDataLine(self, line):
        line=line.strip()
        line = line.replace('|',' ')
        line = line.replace(';',' ')
        line = line.replace('!',' ')
        line = line.replace(':',' ')
        args=line.split()
        for n, i in enumerate(args):
            if i in ['9.9990e+003','-9999.9900','-1.0000e+007','-99999','-99999.0',
                  '-99999.00','-99999.000','-9.9990e+003','999.000', '-9999.99', '999.999',
                  '99','9.999','99.999', '999.9']:
                args[n] = '0.1'       
        if len(line.split()) == 0:
            return
 
        zz  = round(abs(float(args[self._columns[1]])),4)
        dz = round(self.z - zz,4)
        qc = round(float(args[self._columns[2]]),4)  

        slope    =  0.0104
        intercept=  0.0190
                         
        try:
            pw = float(args[self._columns[3]]) 
            if pw<-10:
                pw=0.1
            elif pw>5:
                pw=slope*qc+intercept             
                
        except KeyError: #Deze regel maakt van een tweekolommer een driekolommer
            pw=slope*qc+intercept

        self.dz.append(dz)
        self.qc.append(qc)
        self.pw.append(pw)
        if qc<=0.001:
            qc=0.1
            self.wg.append(10.)
        else:
            wg = abs((pw / qc) * 100.)
        wg = abs((pw / qc) * 100.)

##############################################K-waarde        
        if wg>=0.0:
            if wg >5: wg=15
            ke=math.exp(wg)
        if ke <=0:  ke=1
        else:
            kb  = (qc / ke)*2
            self.kb.append(kb)
            
    def asNumpy(self):
        return np.transpose(np.array([self.dz, self.kb]))

    def asDataFrame(self):
        a = self.asNumpy()
        return pd.DataFrame(data=a, columns=['depth', 'k'])
        
    def plt(self, filename):
        df = self.asDataFrame()
        df = df.sort_values('depth', ascending=False)

        if df.empty:
            return df
        
        df = df.rolling(10).mean() 
        df = df.iloc[:: 10]
        df = df.dropna()
        df = df.reset_index(drop=True)
        df.iloc[-1,1] = 60
        eind = df.iloc[-1,0]
        
        df['k'] = df['k']*1

        dzend = -23
        OZaq  = -40
        
        df = pd.concat([df, pd.DataFrame.from_records([{'depth': OZaq,'k': 0.5}])], ignore_index=True)
        dfn = pd.concat([df, pd.DataFrame.from_records([{'depth':dzend},])], ignore_index=True)
        
        df['kzkh']= 2/(33.653*np.exp(-0.066*(df['k'])))
        df['kzkh']= np.where(df['kzkh']>1, 1, df['kzkh'])
        
        print(df)

        dfd=df['depth']
        dfd.loc['ld']= dzend
        dfn = df.iloc[: , [1]].copy()   
        dfn = pd.concat([df, pd.DataFrame.from_records([{'depth':dzend},])], ignore_index=True)

        df['S'] = np.where(df['depth']>  2, 0.01, 3e-5)
        df['S'] = np.where(df['depth']< -7, 1e-6, df['S']) 
        

        df['Cc'] = (df['k']/(24*60*60)*df['kzkh']/df['S'])/5 #

        ml=Model3D(kaq=df['k'], z=dfn[ 'depth'], Saq= df['S'], kzoverkh=df['kzkh'], tmin=1e-5, 
                   tmax=2, M=10)
        
        mat=pd.read_excel('loc_IIpp.xlsx', engine='openpyxl')  
        mat=mat.replace(',','.')
        
        well    =  []
        well    =  mat[mat['Naam'].str.match ('w')]        
        df.to_excel(filename+'.xlsx')
        x1=mat.iloc[0,1]
        y1=mat.iloc[0,2]
        
        ##Onttrekkingsfilter
        tf =  4
        bf = -23
        nwells= len(well)
        tfind = df['depth'].sub(tf).abs().values.argmin()
        bfind = df['depth'].sub(bf).abs().values.argmin() 
        well1 = arange(tfind, bfind, 1)

        for rnum in well.index:
            punt = well['Naam'][rnum]
            x    = well['x'][rnum] 
            y    = well['y'][rnum]
            punt = Well(ml, x,y, tsandQ=[(0, 864),(1500/1440,0)], 
                        rw=0.25, layers=well1)

        IJssel = HeadLineSinkString(ml, xy=[(-650,200),(-650,-200)], tsandh= 'fixed', res=0, wh=20,
                                  layers = np.arange(9,25,1))        


        laag1 = df['depth'].sub(float(4)).abs().values.argmin() 
        laag2 = df['depth'].sub(float(-10)).abs().values.argmin() 
        laag= [laag1, laag2]
        kleuren(len(laag))
        
####################################
        ml.solve()
# ###################################
        onderlaag = abs(OZaq)-abs(dzend)
        
        KD= round((df['k'].sum()/5+onderlaag*df.iloc[-2,1]),0)
        print('KD  =', KD, ' [m2/dag]')
        
        a=np.amin(punt.headinside(1500/1440))
        print('HeadInside = ',round(a,2), ' [m NAP]')

        for la in laag:
            ml.xsection(x1=x1, x2=x1, y1=y1, y2=y1+1000, npoints=2500, t=1500/1440,
                                layers=[la], lw=3, newfig = False)
        leg=plt.legend(round(df.loc[df.index[laag], 'depth'],1), loc=(0.85,0.05), fontsize = 8, title='Filter op [m NAP]')
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
        plt.grid()
        plt.semilogx()
        plt.xlim(0.3, 1000)
        plt.ylim(-0.25, 0.25)

        ext=pd.read_excel('LoksCoor.xlsx', engine='openpyxl')  
        ext=ext.replace(',','.')
        pbl = ext['pb'].to_numpy()
        for p_b in pbl:
            PB   =  ext[ext['pb'] == p_b]
            PB   = PB.astype(str).values.flatten().tolist()
            dist = float(PB[5])
            dh   = float(PB[7])*-1
            kl   = PB[8]
            plt.plot(dist, dh, kl , markersize = 15, alpha=1)

        plt.yticks(np.arange(-1, 1, 0.2))
        plt.title('Verlaging t.g.v pompproef Ulft  KD = '+ str(int(round(KD,0)))+' [m2/dag]')
        plt.xlabel('Afstand tot bron [m]')
        plt.ylabel('Verlaging [m]')
        plt.savefig('Afstand.png', bbox_inches='tight')
        plt.show()
        plt.close()
        
        """
        Hieronder wordt het stijghoogteverloop tijdens de proef nagebouwd. 
        """
        mat=pd.read_excel('LoksCoor.xlsx', engine='openpyxl')  
        mat=mat.replace(',','.')
        pbl = mat['pb'].to_numpy()
        kleuren(len(pbl))
    
        for p_b in pbl:
            PB    =  mat[mat['pb'] == p_b] 
            x = float(PB['X'])
            y = float(PB['Y'])
            t=np.logspace(-4,1,2500) # 
            s1=ml.head(x, y, t) 
            laag = df['depth'].sub(float(PB['ft'])).abs().values.argmin()
            plt.semilogx(t,s1[laag]*-1, lw=3, alpha=0.5)
            plt.semilogy()
        
        
        mat=pd.read_excel('Pb_PP.xlsx', index_col=None, skiprows=0, engine='openpyxl') 
        tijd   = mat.columns.values.tolist()
        punten = list(mat)
        x=mat['tijd']/1440
        kleuren(len(tijd))
        tijd1=tijd[1:]
        if len(tijd[1:])!=len(pbl):
            print('Verschillend aantal meetpunten')
            return

        for ypb in tijd1:
            y1=mat[ypb]/100
            plt.plot(x,y1, '+', markersize=8)
    
        plt.grid(axis='both')
        plt.ylim(1e-3, 0)
        plt.xlabel('tijd [dagen]')
        plt.ylabel('verlaging [m]')
        plt.savefig('Tijd_dh_Log.png', bbox_inches='tight')
        plt.legend(punten[1:])
        plt.show()
        plt.close()
######################
        mat=pd.read_excel('LoksCoor.xlsx', engine='openpyxl')  
        mat=mat.replace(',','.')
        pbl = mat['pb'].to_numpy()
        kleuren(len(pbl))
    
        for p_b in pbl:
            PB    =  mat[mat['pb'] == p_b] 
            x = float(PB['X'])
            y = float(PB['Y'])
            t=np.logspace(-4,1,2500) # 
            s1=ml.head(x, y, t) 
            laag = df['depth'].sub(float(PB['ft'])).abs().values.argmin()
            plt.semilogx(t,s1[laag]*-1, lw=3, alpha=0.5)
        
        
        mat=pd.read_excel('Pb_PP.xlsx', index_col=None, skiprows=0, engine='openpyxl') 
        tijd   = mat.columns.values.tolist()
        punten = list(mat)
        x=mat['tijd']/1440
        kleuren(len(tijd))
        tijd1=tijd[1:]
        if len(tijd[1:])!=len(pbl):
            print('Verschillend aantal meetpunten')
            return

        for ypb in tijd1:
            y1=mat[ypb]/100
            plt.plot(x,y1, '+', markersize=8)
    
        plt.grid(axis='both')
        plt.ylim(0.2,-0.2)
        plt.xlabel('tijd [dagen]')
        plt.ylabel('verlaging [m]')
        plt.savefig('Tijd_dh.png', bbox_inches='tight')
        plt.legend(punten[1:])
        plt.show()
        plt.close()

# ##### Poging tot Cc        
        kleuren(1)
        df= df[df['depth']>eind]
        
        df.plot(x='Cc',  y='depth', color='blue', linewidth=1, alpha=0.3, legend=False, figsize=(3,8))
        plt.fill_betweenx(df['depth'], df['Cc'], color='blue', alpha =0.2)
        plt.semilogx()
        plt.grid(axis='both')
        plt.ylim(-50,10)
        plt.xlim(0.01,100)
        plt.xticks(ticks=[10e-3,10e-2,10e-1,10e0,10e2], labels=['0.01','0.1','1','10','100'])
        plt.yticks(np.arange(-50,20,2))
        plt.xlabel('Consolidatie coefficient')
        plt.ylabel('Diepte [m NAP]')
        plt.savefig('Cc', bbox_inches='tight')
        plt.show()
        plt.close()
 
        
for filename in os.listdir(os.getcwd()):
    if filename.endswith ('.GEF') or filename.endswith ('.gef'):
        if __name__=="__main__":
            g=GEF()
            g.readFile(filename)
            g.plt(filename)
