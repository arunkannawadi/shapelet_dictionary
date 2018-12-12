def visualise_signal(signal,true_signal=None,**kwargs):
  import matplotlib
  import matplotlib.pyplot as plt
  import matplotlib.cm as cm
  import numpy as np

  if true_signal is not None:
      fig,ax = plt.subplots(2,3)
  else:
      fig,ax = plt.subplots(2,2)

  if ax.shape==(2,3):
      ax[0,1].imshow(true_signal)
      bins = 100
      ax[1,1].hist(signal.residual.flatten(),bins=bins,normed=True)

  ax[0,0].imshow(signal.image)
  ax[0,0].set_title('Observed image')

  ax[1,0].imshow(signal.signal)
  ax[1,0].set_title('Reconstructed image')

  ax[0,-1].imshow(signal.residual)
  ax[0,-1].set_title('Residual image')

  plt.show()
