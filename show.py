show2D([fbp_recon, sirt_nz.solution, cgls.solution, spdhg.solution],
      ['fbp', 'sirt', 'cgls', 'spdhg TV'],
      cmap=cmap, fix_range=(0,.004),
      num_cols=2)

plt.semilogy(np.concatenate(([0],np.sort(np.cumsum(sirt_nz3.timing)[-1:0:-sirt_nz3.update_objective_interval]))), 
             sirt_nz3.objective, label='SIRT')
plt.semilogy(np.concatenate(([0],np.sort(np.cumsum(cgls3.timing)))), 
             cgls3.objective, label='CGLS')
#plt.semilogy(np.concatenate(([0],np.sort(np.cumsum(myFISTATV.timing)[-1:0:-myFISTATV.update_objective_interval]))),
             #myFISTATV.objective, label='FISTATV')
#plt.semilogy(np.concatenate(([0],np.sort(np.cumsum(pdhg.timing)[-1:0:-pdhg.update_objective_interval]))), \
#             pdhg.objective, label='PDHGTV')
plt.semilogy(np.concatenate(([0],np.sort(np.cumsum(spdhg3.timing)[-1:0:-spdhg3.update_objective_interval]))), 
             spdhg3.objective, label='SPDHGTV')

plt.xlabel('Timing')
plt.ylabel('Objective function')
plt.legend()


plt.semilogy(np.arange(0,sirt_nz3.iteration+sirt_nz3.update_objective_interval, sirt_nz3.update_objective_interval),
             sirt_nz3.objective, label='SIRT')
plt.semilogy(np.arange(0,cgls3.iteration+cgls3.update_objective_interval, cgls3.update_objective_interval),
             cgls3.objective, label='CGLS')
#plt.semilogy(np.arange(0,myFISTATV.iteration+myFISTATV.update_objective_interval, myFISTATV.update_objective_interval), 
#             myFISTATV.objective, label='FISTATV')
plt.semilogy(np.arange(0,spdhg3.iteration+spdhg3.update_objective_interval, spdhg3.update_objective_interval), 
             spdhg3.objective, label='SPDHGTV')

plt.xlabel('Epochs')
plt.ylabel('Objective function')
plt.legend()


#### TV denoising?
alpha = 0.012
TV = alpha * TotalVariation(max_iteration=7)
fbpTV = TV.proximal(fbp_recon, tau=0.1)
show2D([fbp_recon_s800, fbpTV,fbp_recon], 
       title=['fbp full','Total variation', 'fbp reduced'], 
       origin="upper", num_cols=3,
      cmap = cmap, fix_range=(0,.004))


### Histogram
fig,ax=plt.subplots(1,2,figsize=(15,5))
ax[0].hist(np.sort(fbp_recon_s800.as_array().ravel())[0:-1],bins=400);
ax[1].hist(np.sort(sirt_nz.solution.as_array().ravel())[0:-1],
           bins=800);


#### multiple in one hist
x1 = np.sort(fbp_recon_s800.as_array().ravel())
x2 = np.sort(sirt_nz.solution.as_array().ravel())
x3 = np.sort(s800_clean.as_array().ravel())

bins = np.linspace(-0.01, 0.08, 1000)

plt.hist(x1, bins, alpha=0.5, label='fbp')
plt.hist(x2, bins, alpha=0.5, label='sirt')
plt.hist(x3, bins, alpha=0.5, label='orginal')
#plt.ylim = (0,5000)
plt.legend(loc='upper right')
plt.show()