import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

#modulation
def bpsk_mod(bits):
  return 2*bits-1

def bpsk_demod(symbol):
  return (symbol>=0).astype(int)

#channels
def awgn_channel(x,EbN0_db):
  EbN0=10**(EbN0_db/10)
  N0=1/EbN0
  noise=np.sqrt(N0/2)*np.random.randn(len(x))
  return x+noise

def rayleigh_channel(x,EbN0_db):
  EbN0=10**(EbN0_db/10)
  N0=1/EbN0
  h=np.sqrt(0.5)*np.random.randn(len(x))
  y=h*x
  noise=np.sqrt(N0/2)*np.random.randn(len(x))
  return y+noise,h

#repetition codes
def rep_encode(bits,n):
  return np.repeat(bits,n)

def rep_decode(bits,n):
  bits=bits.reshape(-1,n)
  return (np.sum(bits,axis=1)>=(n/2)).astype(int)

#hamming (7,4)
G74 = np.array([[1,0,0,0,1,1,0],
[0,1,0,0,1,0,1],
[0,0,1,0,0,1,1],
[0,0,0,1,1,1,1]])
H74 = np.array([[1,1,0,1,1,0,0],
[1,0,1,1,0,1,0],
[0,1,1,1,0,0,1]])

#hamming (15,11)
G1511 = np.hstack((np.eye(11), np.ones((11,4),dtype=int)))
H1511 = np.hstack((np.ones((4,11),dtype=int), np.eye(4)))

#hamming functions
def hamming_encode(bits, G, k):
    coded=[]
    bits=bits.reshape(-1, k)
    for block in bits:
        codeword=(block @ G) % 2
        coded.extend(codeword)
    return np.array(coded)

def hamming_decode(bits,H,k):
    n=H.shape[1]
    bits=bits.reshape(-1,n)
    decoded=[]
    for codeword in bits:
        syndrome=(H @ codeword) % 2
        if not np.all(syndrome==0):
            for i in range(n):
                if np.array_equal(H[:,i],syndrome):
                    codeword[i]^= 1
                    break
        decoded.extend(codeword[:k])
    return np.array(decoded)

#Ber evalution
def ber_sim(EbN0_range,channel,coding):
  N=10000
  ber=[]
  for snr in EbN0_range:
    bits=np.random.randint(0,2,N)
    if coding=="none":
      tx=bpsk_mod(bits)
      if channel=="awgn":
        rx=awgn_channel(tx,snr)
      elif channel=="rayleigh":
        rx,h=rayleigh_channel(tx,snr)
        rx=rx/h
      bits_hat=bpsk_demod(rx)
    elif coding == 'rep3':
      c=rep_encode(bits,3)
      tx=bpsk_mod(c)/np.sqrt(3)
      rx,h=rayleigh_channel(tx,snr)
      rx=rx/h
      bh=bpsk_demod(rx)
      bits_hat=rep_decode(bh,3)

    elif coding == 'rep5':
      c=rep_encode(bits,5)
      tx=bpsk_mod(c)/np.sqrt(5)
      rx,h=rayleigh_channel(tx,snr)
      rx=rx/h
      bh=bpsk_demod(rx)
      bits_hat=rep_decode(bh,5)
    elif coding == 'ham74':
      c = hamming_encode(bits[:len(bits)//4*4],G74,4)
      tx = bpsk_mod(c)/np.sqrt(7/4)
      rx,h = rayleigh_channel(tx,snr)
      rx = rx/h
      bh = bpsk_demod(rx)
      bits_hat = hamming_decode(bh,H74,4)
    elif coding == 'ham1511':
      c = hamming_encode(bits[:len(bits)//11*11],G1511,11)
      tx = bpsk_mod(c)/np.sqrt(15/11)
      rx,h = rayleigh_channel(tx,snr)
      rx = rx/h
      bh = bpsk_demod(rx)
      bits_hat = hamming_decode(bh,H1511,11)

    ber.append(np.mean(bits[:len(bits_hat)]!=bits_hat))
  return ber

class Gui:
    def __init__(self, root):
        self.root=root
        root.title('BER Performance of Coded and Uncoded BPSK Systems')

        ttk.Label(root,text='Eb/N0 Range(0,2,4,6)').pack()
        self.snr=ttk.Entry(root)
        self.snr.pack()

        #channel check boxes
        ttk.Label(root,text='Channel').pack()

        self.awgn=tk.BooleanVar()
        self.rayleigh=tk.BooleanVar()

        ttk.Checkbutton(root,text='AWGN',variable=self.awgn).pack()
        ttk.Checkbutton(root,text='Rayleigh',variable=self.rayleigh).pack()

        #coding check boxes
        ttk.Label(root,text='Coding').pack()

        self.none=tk.BooleanVar(value=True)
        self.rep3=tk.BooleanVar()
        self.rep5=tk.BooleanVar()
        self.ham74=tk.BooleanVar()
        self.ham1511=tk.BooleanVar()

        ttk.Checkbutton(root,text='No Coding',variable=self.none).pack()
        ttk.Checkbutton(root,text='Repetition 1/3',variable=self.rep3).pack()
        ttk.Checkbutton(root,text='Repetition 1/5',variable=self.rep5).pack()
        ttk.Checkbutton(root,text='Hamming (7,4)',variable=self.ham74).pack()
        ttk.Checkbutton(root,text='Hamming (15,11)',variable=self.ham1511).pack()

        ttk.Button(root,text='Run',command=self.run).pack()
        self.out=ttk.Label(root,text='')
        self.out.pack()

    def run(self):
        try:
            EbN0=[float(x) for x in self.snr.get().split(',')]
        except:
            self.out.config(
                text='Error: Enter Eb/N0 as comma separated values (e.g. 0,2,4,6)'
            )
            return

        plotted_any = False
        message=[]
        plots=[]
        if self.awgn.get():
            if self.none.get():
                plots.append(('awgn','none','AWGN - No Coding'))
            else:
                message.append(
                    'AWGN with channel coding is not required by the project and was not plotted.'
                )


        if self.rayleigh.get():
            if self.none.get():
                plots.append(('rayleigh','none','Rayleigh - No Coding'))
            if self.rep3.get():
                plots.append(('rayleigh','rep3','Rayleigh - Rep 1/3'))
            if self.rep5.get():
                plots.append(('rayleigh','rep5','Rayleigh - Rep 1/5'))
            if self.ham74.get():
                plots.append(('rayleigh','ham74','Rayleigh - Hamming (7,4)'))
            if self.ham1511.get():
                plots.append(('rayleigh','ham1511','Rayleigh - Hamming (15,11)'))

        if not plots:
            self.out.config(text='No valid case selected.')
            return

        plt.figure()

        for ch, cd, label in plots:
            ber = ber_sim(EbN0, ch, cd)
            plt.semilogy(EbN0, ber, marker='o', label=label)
            plotted_any = True

        plt.xlabel('Eb/N0 (dB)')
        plt.ylabel('BER')
        plt.grid(True)
        plt.legend()
        plt.title('BER vs Eb/N0')
        plt.show()

        if message:
            self.out.config(text=' '.join(message))
        else:
            self.out.config(text='Selected cases plotted.')


root=tk.Tk()
Gui(root)
root.mainloop()