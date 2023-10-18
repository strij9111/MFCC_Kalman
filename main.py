import numpy as np
import os
import sys
import librosa
import soundfile as sf
from tqdm import tqdm


class ErrorCovarianceMatrix:

    def __init__(self,p,q):

        self.p = p

        self.q = q

        # initialize matrices

        self.mqq = np.zeros((q,q), dtype=np.float32)
        self.mqp = np.zeros((q,p), dtype=np.float32)
        self.mpq = np.zeros((p,q), dtype=np.float32)
        self.mpp = np.zeros((p,p), dtype=np.float32)

        self.oqq = np.asarray([0,0], dtype=np.int32)
        self.oqp = np.asarray([0,0], dtype=np.int32)
        self.opq = np.asarray([0,0], dtype=np.int32)
        self.opp = np.asarray([0,0], dtype=np.int32)

    def getqq(self,i,j):

        return self.mqq[(self.oqq[0]+i) % self.q,(self.oqq[1]+j) % self.q]

    def getqp(self, i, j):

        return self.mqp[(self.oqp[0] + i) % self.q, (self.oqp[1] + j) % self.p]

    def getpq(self, i, j):

        return self.mpq[(self.opq[0]+i) % self.p, (self.opq[1] + j) % self.q]

    def getpp(self, i, j):

        return self.mpp[(self.opp[0]+i) % self.p, (self.opp[1]+j) % self.p]

    def setqq(self,i,j,v):

        self.mqq[(self.oqq[0] + i) % self.q, (self.oqq[1] + j) % self.q] = v

    def setqp(self, i, j, v):

        self.mqp[(self.oqp[0] + i) % self.q, (self.oqp[1] + j) % self.p] = v

    def setpq(self, i, j, v):

        self.mpq[(self.opq[0] + i) % self.p, (self.opq[1] + j) % self.q] = v

    def setpp(self, i, j, v):

        self.mpp[(self.opp[0] + i) % self.p, (self.opp[1] + j) % self.p] = v

    def setC(self,C):

        self.mqq = np.float32(C[:self.q,:self.q])
        self.mqp = np.float32(C[:self.q,self.q:])
        self.mpq = np.float32(C[self.q:,:self.q])
        self.mpp = np.float32(C[self.q:,self.q:])

    def get(self):

        C = np.zeros((self.q+self.p,self.q+self.p), dtype=np.float32)

        for i in range(self.q):
            for j in range(self.q):
                C[i,j] = self.getqq(i,j)

        for i in range(self.q):
            for j in range(self.p):
                C[i,self.q+j] = self.getqp(i,j)

        for i in range(self.p):
            for j in range(self.q):
                C[self.q+i,j] = self.getpq(i,j)

        for i in range(self.p):
            for j in range(self.p):
                C[self.q+i,self.q+j] = self.getpp(i,j)

        return C

    def FCT(self, asp, ans):

        # qq

        for i in range(self.q):

            accu = 0

            for j in range(self.q):

                accu -= ans[j]*self.getqq(i,j)

            self.setqq(i, self.q, accu)

        self.oqq[1] = (self.oqq[1] + 1) % self.q

        # qp

        for i in range(self.p):

            accu = 0

            for j in range(self.q):

                accu -= ans[j] * self.getpq(i, j)

            self.setpq(i, self.q, accu)

        self.opq[1] = (self.opq[1] + 1) % self.q

        # pq

        for i in range(self.q):

            accu = 0

            for j in range(self.p):

                accu -= asp[j] * self.getqp(i, j)

            self.setqp(i, self.p, accu)

        self.oqp[1] = (self.oqp[1] + 1) % self.p

        # pp

        for i in range(self.p):

            accu = 0

            for j in range(self.p):

                accu -= asp[j] * self.getpp(i, j)

            self.setpp(i, self.p, accu)

        self.opp[1] = (self.opp[1] + 1) % self.p


    def FC(self, asp, ans):

        # qq

        for i in range(self.q):

            accu = 0

            for j in range(self.q):

                accu -= ans[j]*self.getqq(j,i)

            self.setqq(self.q,i,accu)

        self.oqq[0] = (self.oqq[0]+1)%self.q

        # qp

        for i in range(self.p):

            accu = 0

            for j in range(self.q):

                accu -= ans[j] * self.getqp(j, i)

            self.setqp(self.q, i, accu)

        self.oqp[0] = (self.oqp[0] + 1) % self.q

        # pq

        for i in range(self.q):

            accu = 0

            for j in range(self.p):

                accu -= asp[j] * self.getpq(j, i)

            self.setpq(self.p, i, accu)

        self.opq[0] = (self.opq[0] + 1) % self.p

        # pp

        for i in range(self.p):

            accu = 0

            for j in range(self.p):

                accu -= asp[j] * self.getpp(j, i)

            self.setpp(self.p, i, accu)

        self.opp[0] = (self.opp[0] + 1) % self.p

    def predictor(self,asp, ans, sig_s, sig_n):

        self.FC(asp,ans)
        self.FCT(asp,ans)
        qq = self.getqq(self.q-1,self.q-1) + sig_n*sig_n
        pp = self.getpp(self.p-1,self.p-1) + sig_s*sig_s
        self.setqq(self.q-1,self.q-1,qq)
        self.setpp(self.p-1,self.p-1,pp)

    def gain(self):

        g = np.zeros((self.p+self.q,), np.float32)

        for i in range(self.q):

            g[i] = self.getqq(i,self.q-1) + self.getqp(i,self.p-1)

        for i in range(self.p):

            g[self.q+i] = self.getpq(i, self.q-1) + self.getpp(i, self.p-1)

        g /= (self.getqq(self.q-1,self.q-1)+self.getqp(self.q-1,self.p-1)+self.getpp(self.p-1,self.p-1) + self.getpq(self.p-1,self.q-1))

        return g

    def update(self,gain):

        auxqq = np.zeros((self.q,self.q), dtype=np.float32)
        auxqp = np.zeros((self.q,self.p), dtype=np.float32)
        auxpq = np.zeros((self.p,self.q), dtype=np.float32)
        auxpp = np.zeros((self.p,self.p), dtype=np.float32)

        for i in range(self.q):

            for j in range(self.q):

                v = self.getqq(i,j) - gain[i]*(self.getqq(self.q-1,j)+self.getpq(self.p-1,j))

                auxqq[i,j] = v

        for i in range(self.q):

            for j in range(self.p):

                v = self.getqp(i,j) - gain[i]*(self.getqp(self.q-1,j) + self.getpp(self.p-1,j))

                auxqp[i,j] = v

        for i in range(self.p):

            for j in range(self.q):

                v = self.getpq(i,j) - gain[self.q+i]*(self.getpq(self.p-1,j)+self.getqq(self.q-1,j))

                auxpq[i,j] = v

        for i in range(self.p):

            for j in range(self.p):

                v = self.getpp(i,j) - gain[self.q+i]*(self.getpp(self.p-1,j) + self.getqp(self.q-1,j))

                auxpp[i,j] = v

        """================"""

        for i in range(self.q):

            for j in range(self.q):

                self.setqq(i,j,auxqq[i,j])

        for i in range(self.q):

            for j in range(self.p):

                self.setqp(i,j,auxqp[i,j])

        for i in range(self.p):

            for j in range(self.q):

                self.setpq(i,j,auxpq[i,j])

        for i in range(self.p):

            for j in range(self.p):

                self.setpp(i,j,auxpp[i,j])


class StateVector:

    def __init__(self,p,q):

        self.xq = np.zeros((q,), dtype=np.float32)
        self.xp = np.zeros((p,), dtype=np.float32)

        self.q = q
        self.p = p

        self.oq = 0
        self.op = 0

    def getq(self,i):

        return self.xq[(self.oq+i)%self.q]

    def getp(self,i):

        return self.xp[(self.op+i)%self.p]

    def setq(self,i,v):

        self.xq[(self.oq+i)%self.q] = v

    def setp(self,i,v):

        self.xp[(self.op+i)%self.p] = v

    def setX(self,x):

        self.xq = x[:self.q]

        self.xp = x[self.q:]

    def predictor(self,asp,ans):

        accu = 0

        for i in range(self.q):

            accu -= self.getq(i)*ans[i]

        self.setq(self.q, accu)

        self.oq = (self.oq+1)%self.q

        accu = 0

        for i in range(self.p):

            accu -= self.getp(i)*asp[i]

        self.setp(self.p,accu)

        self.op = (self.op+1)%self.p

    def update(self,gain,rt):

        aux = self.getq(self.q-1)+self.getp(self.p-1)

        aux = rt-aux

        ga = gain*aux

        for i in range(self.q):

            v = self.getq(i)+ga[i]

            self.setq(i,v)

        for i in range(self.p):

            v = self.getp(i) + ga[self.q+i]

            self.setp(i,v)

    def get(self):

        x = np.zeros((self.p+self.q,), dtype=np.float32)

        for i in range(self.q):

            x[i] = self.getq(i)

        for i in range(self.p):

            x[self.q+i] = self.getp(i)

        return x
        
    
class STPKalman:

    def __init__(self, p=10, q=1):

        self.p = p
        self.q = q

        self.ecm = ErrorCovarianceMatrix(p,q)
        self.sv = StateVector(p,q)

    def step(self,rt,asp,ans,sig_s,sig_n):
        """
        :param asp: Параметры речи в STP, анализирующий полином имеет вид: \(1 + \text{asp}[p-1] \cdot z^{-1} + \ldots + \text{asp}[0] \cdot z^{-p}\)
        :param ans: Параметры шума в STP, анализирующий полином аналогичен
        :return: Фильтрованный вектор состояний; первые \(q\) элементов для шума, следующие \(p\) элементов для речи
        """

        asp = asp.astype(np.float32)
        ans = ans.astype(np.float32)
        sig_s = np.float32(sig_s)
        sig_n = np.float32(sig_n)

        self.sv.predictor(asp,ans)
        self.ecm.predictor(asp,ans,sig_s,sig_n)
        G = self.ecm.gain()
        self.sv.update(G,rt)
        self.ecm.update(G)

        return self.sv.getp(0)

SAMPLING_RATE = 16000

if __name__ == "__main__":

    signal, sr = librosa.load("orig.wav", sr=None)
    noise, sr = librosa.load("noise.wav", sr=None)
    signal = signal[16000:64000]
    noise = noise[:len(signal)]

    SNR = 10.

    noise *= (np.linalg.norm(signal)**2/np.linalg.norm(noise)**2/10**(SNR/10.))**.5

    mixed = signal + noise

    start = 0
    frame = 320 # размер кадра
    n_mfcc = 80
    p = n_mfcc
    q = 32

    kl = STPKalman(p, q)

    e = np.zeros_like(mixed)

    num_frames = len(mixed) // frame
    pbar = tqdm(total=num_frames)
    
    while start+frame <= len(mixed):
        fs = signal[start:start+frame]
        fn = noise[start:start+frame]
        fr = mixed[start:start+frame]

        asp = librosa.feature.mfcc(y=fs, sr=sr, n_mfcc=n_mfcc, n_fft=256, hop_length=64)
        asp = np.mean(asp, axis=1)
        asp_mean = np.mean(asp)
        asp_std = np.std(asp)
        asp = (asp - asp_mean) / asp_std
        sig_s = np.var(asp)

        ans = librosa.feature.mfcc(y=fn, sr=sr, n_mfcc=q, n_fft=256, hop_length=64)
        ans = np.mean(ans, axis=1)
        ans_mean = np.mean(ans)
        ans_std = np.std(ans)
        ans = (ans - ans_mean) / ans_std
        sig_n = np.var(ans)

        for i in range(frame):

            e[start + i] = kl.step(fr[i], asp[::-1], ans[::-1], sig_s, sig_n)

        start += frame
        pbar.update(1)
    
    pbar.close()  
    sf.write('output.wav', e, sr)
    sf.write('output_mixed.wav', mixed, sr)



